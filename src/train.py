"""
Script de entrenamiento de AMFuse sobre CMU-MOSEI.

Uso:
    cd amfuse_2
    python src/train.py
    python src/train.py --epochs 30 --lr 1e-4 --batch_size 64 --missing_rate 0.2
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_loaders
from evaluate import compute_metrics
from logger import Logger
from model import AMFuse


# Paths por defecto relativos al archivo del script, no al CWD.
# Así funciona igual desde VS Code, terminal, o cualquier directorio.
_SRC_DIR  = Path(__file__).resolve().parent          # .../amfuse_2/src
_ROOT_DIR = _SRC_DIR.parent                          # .../amfuse_2
_DEFAULT_DATA = str(_ROOT_DIR / "data")
_DEFAULT_EXP  = str(_ROOT_DIR / "experiments")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entrenar AMFuse en CMU-MOSEI")
    p.add_argument("--data_dir",     default=_DEFAULT_DATA, type=str)
    p.add_argument("--exp_dir",      default=_DEFAULT_EXP,  type=str)
    p.add_argument("--task",         default="regression",
                   choices=["regression", "classification"])
    p.add_argument("--epochs",       default=30,   type=int)
    p.add_argument("--batch_size",   default=32,   type=int)
    p.add_argument("--lr",           default=1e-4, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--d_hidden",     default=256,  type=int)
    p.add_argument("--missing_rate", default=0.0,  type=float,
                   help="Probabilidad de eliminar cada modalidad en entrenamiento")
    p.add_argument("--num_workers",  default=0,    type=int)
    p.add_argument("--seed",         default=42,   type=int)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss(task: str):
    if task == "regression":
        return nn.L1Loss()
    return nn.CrossEntropyLoss()


def label_to_class(labels: torch.Tensor) -> torch.Tensor:
    cls = torch.zeros_like(labels, dtype=torch.long)
    cls[labels > 0]  = 2
    cls[(labels >= -1) & (labels <= 0)] = 1
    cls[labels < -1] = 0
    return cls


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, task, log,
                epoch, show_first_batch_verbose):
    """Entrena una época. Muestra verbose del primer batch si se solicita."""
    model.train()
    total_loss = 0.0
    first_batch = True

    pbar = tqdm(loader, desc=f"  Entrenando época {epoch}", leave=False,
                unit="batch", dynamic_ncols=True)

    for batch in pbar:
        text   = batch["text"].to(device)
        audio  = batch["audio"].to(device)
        video  = batch["video"].to(device)
        labels = batch["label"].to(device)
        mt     = batch["mask_t"].to(device)
        ma     = batch["mask_a"].to(device)
        mv     = batch["mask_v"].to(device)

        # Verbose solo para el primer batch de la época indicada
        if first_batch and show_first_batch_verbose:
            model.set_verbose(log)
            log.subsection(f"Detalle módulos — Época {epoch}, Batch 1")
        else:
            model.set_silent()

        optimizer.zero_grad()
        out, aux = model(text, audio, video, mt, ma, mv)

        if task == "regression":
            loss = loss_fn(out.squeeze(-1), labels)
        else:
            loss = loss_fn(out, label_to_class(labels))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if first_batch and show_first_batch_verbose:
            model.set_silent()
        first_batch = False

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, task, split_name="valid"):
    model.eval()
    model.set_silent()
    total_loss = 0.0
    preds, gts = [], []
    conf_t_all, conf_a_all, conf_v_all = [], [], []

    pbar = tqdm(loader, desc=f"  Evaluando {split_name}", leave=False,
                unit="batch", dynamic_ncols=True)

    for batch in pbar:
        text   = batch["text"].to(device)
        audio  = batch["audio"].to(device)
        video  = batch["video"].to(device)
        labels = batch["label"].to(device)
        mt     = batch["mask_t"].to(device)
        ma     = batch["mask_a"].to(device)
        mv     = batch["mask_v"].to(device)

        out, aux = model(text, audio, video, mt, ma, mv)
        conf_t_all.append(aux["c_t"].cpu())
        conf_a_all.append(aux["c_a"].cpu())
        conf_v_all.append(aux["c_v"].cpu())

        if task == "regression":
            loss = loss_fn(out.squeeze(-1), labels)
            preds.append(out.squeeze(-1).cpu())
            gts.append(labels.cpu())
        else:
            cls  = label_to_class(labels)
            loss = loss_fn(out, cls)
            preds.append(out.argmax(dim=-1).cpu())
            gts.append(cls.cpu())

        total_loss += loss.item() * labels.size(0)

    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    metrics = compute_metrics(preds, gts, task)

    conf_means = {
        "c_t": torch.cat(conf_t_all).mean().item(),
        "c_a": torch.cat(conf_a_all).mean().item(),
        "c_v": torch.cat(conf_v_all).mean().item(),
    }
    return total_loss / len(loader.dataset), metrics, conf_means


# ── Stats cada N épocas ───────────────────────────────────────────────────────

def print_model_stats(model, epoch, log):
    """Imprime normas de gradientes y parámetros por módulo."""
    log.subsection(f"Stats del modelo — Época {epoch}")

    modules = {
        "M2  Proyectores":    model.projectors,
        "M3A Generador":      model.generator,
        "M3B Confiabilidad":  model.confidence,
        "M4  CrossAttn":      model.cross_attn,
        "M5  Fusión":         model.fusion,
        "M6  Clasificador":   model.classifier,
    }

    log("  Normas de gradientes (promedio por módulo):")
    grad_norms = model.grad_norms()
    for raw_name, norm in grad_norms.items():
        pretty = raw_name.replace("_", " ")
        log(f"    {pretty:<25s}: {norm:.6f}")

    log("")
    log("  Normas de parámetros (W, media ± std):")
    for label, mod in modules.items():
        w_norms = [p.data.norm().item() for p in mod.parameters() if p.ndim >= 2]
        if w_norms:
            mean_n = np.mean(w_norms)
            std_n  = np.std(w_norms)
            log(f"    {label:<25s}: {mean_n:.4f} ± {std_n:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)

    # Directorio de experimento con timestamp
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.exp_dir) / run_ts
    exp_dir.mkdir(parents=True, exist_ok=True)

    log    = Logger(exp_dir / "run.log")
    writer = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))

    # ── Banner inicial ───────────────────────────────────────────────────────
    log.section("AMFuse — Inicio de entrenamiento")
    log.timestamp("Fecha/hora:")
    log("")

    log("  Configuración:")
    for k, v in vars(args).items():
        log.kv(k, v)

    log("")
    log("  Archivos de salida:")
    log.kv("Directorio experimento", str(exp_dir))
    log.kv("Log detallado",          str(exp_dir / "run.log"))
    log.kv("Mejor checkpoint",       str(exp_dir / "best_model.pt"))
    log.kv("Último checkpoint",      str(exp_dir / "last_model.pt"))
    log.kv("Métricas JSON",          str(exp_dir / "metrics.json"))
    log.kv("TensorBoard",            str(exp_dir / "tensorboard"))

    # ── Dispositivo ──────────────────────────────────────────────────────────
    log.subsection("Dispositivo")
    device = torch.device(args.device)
    log.kv("Device", str(device))
    if device.type == "cuda":
        log.kv("GPU", torch.cuda.get_device_name(0))
        log.kv("CUDA version", torch.version.cuda)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.kv("VRAM total", f"{mem:.1f} GB")

    # ── Datos ────────────────────────────────────────────────────────────────
    log.subsection("Carga de datos")
    log("  Cargando splits de CMU-MOSEI ...")
    t0 = time.time()
    loaders = get_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        missing_rate=args.missing_rate,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    log(f"  Splits cargados en {time.time()-t0:.1f}s")
    log("")
    log("  Tamaño de splits:")
    for split, loader in loaders.items():
        log.kv(f"  {split}", f"{len(loader.dataset):,} muestras  "
                             f"({len(loader)} batches × batch_size={args.batch_size})")
    log("")
    log("  Dimensiones de features:")
    log.kv("  Texto (BERT CLS)", "768")
    log.kv("  Audio (COVAREP)",  "74")
    log.kv("  Video (FACET)",    "35")
    log.kv("  Missing rate train", f"{args.missing_rate*100:.0f}%")

    # ── Modelo ───────────────────────────────────────────────────────────────
    log.subsection("Arquitectura AMFuse")
    model = AMFuse(task=args.task, d=args.d_hidden).to(device)
    total, trainable = model.count_params()

    log(f"  Pipeline: M1(estado) -> M2(proyección) -> M3A(generación) -> "
        f"M3B(confiabilidad) -> M4(cross-attn) -> M5(fusión) -> M6(clasificador)")
    log("")
    log("  Parámetros por módulo:")

    module_map = {
        "M2  UnimodalProjectors":  model.projectors,
        "M3A MissingGenerator":    model.generator,
        "M3B ConfidenceEstimator": model.confidence,
        "M4  DynamicCrossAttn":    model.cross_attn,
        "M5  ConfidenceAwareFusion": model.fusion,
        "M6  Classifier":          model.classifier,
    }
    for label, mod in module_map.items():
        n = sum(p.numel() for p in mod.parameters())
        log(f"    {label:<30s}: {n:>10,}")
    log(f"    {'TOTAL':<30s}: {total:>10,}")
    log(f"    {'ENTRENABLES':<30s}: {trainable:>10,}")

    log("")
    log("  Hiperparámetros del modelo:")
    log.kv("  Dimensión latente d",  args.d_hidden)
    log.kv("  Cabezas de atención",  8)
    log.kv("  Beta (penalización)",  0.5)
    log.kv("  Task",                 args.task)

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    log.subsection("Optimizador")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = build_loss(args.task)

    log.kv("Optimizador",        "AdamW")
    log.kv("Learning rate",      args.lr)
    log.kv("Weight decay",       args.weight_decay)
    log.kv("Scheduler",          f"CosineAnnealingLR (T_max={args.epochs})")
    log.kv("Loss",               "MAE (L1Loss)" if args.task == "regression" else "CrossEntropyLoss")
    log.kv("Gradient clipping",  "max_norm=1.0")

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    log.section(f"Entrenamiento — {args.epochs} épocas")

    best_val_loss  = float("inf")
    best_epoch     = 0
    best_ckpt      = exp_dir / "best_model.pt"
    last_ckpt      = exp_dir / "last_model.pt"
    history        = []   # lista de dicts con stats por época
    total_start    = time.time()

    # Header de la tabla de épocas
    log("")
    if args.task == "regression":
        header = f"  {'Época':>5}  {'Tr Loss':>8}  {'Val Loss':>8}  " \
                 f"{'MAE':>7}  {'Corr':>7}  {'Acc7':>7}  {'Acc2':>7}  " \
                 f"{'F1':>7}  {'Tiempo':>7}  {'Mejor'}"
    else:
        header = f"  {'Época':>5}  {'Tr Loss':>8}  {'Val Loss':>8}  " \
                 f"{'Acc3':>7}  {'F1_w':>7}  {'Tiempo':>7}  {'Mejor'}"
    log(header)
    log("  " + "-" * (len(header) - 2))

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Mostrar verbose del primer batch en la época 1, 5, 10, ... o la última
        show_verbose = (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs)

        # ── Entrenamiento ────────────────────────────────────────────────────
        tr_loss = train_epoch(
            model, loaders["train"], optimizer, loss_fn,
            device, args.task, log, epoch, show_verbose
        )

        # ── Validación ───────────────────────────────────────────────────────
        val_loss, val_met, val_conf = eval_epoch(
            model, loaders["valid"], loss_fn, device, args.task, "val"
        )
        scheduler.step()
        elapsed = time.time() - epoch_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_met,
                "args": vars(args),
            }, best_ckpt)

        # Guardar siempre el último checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }, last_ckpt)

        # ── Fila de tabla por época ──────────────────────────────────────────
        marker = " << MEJOR" if is_best else ""
        if args.task == "regression":
            row = (
                f"  {epoch:>5}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
                f"{val_met['MAE']:>7.4f}  {val_met['Corr']:>7.4f}  "
                f"{val_met['Acc7']:>7.4f}  {val_met['Acc2']:>7.4f}  "
                f"{val_met['F1']:>7.4f}  {elapsed:>6.1f}s{marker}"
            )
        else:
            row = (
                f"  {epoch:>5}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
                f"{val_met['Acc3']:>7.4f}  {val_met['F1_w']:>7.4f}  "
                f"{elapsed:>6.1f}s{marker}"
            )
        log(row)

        # ── TensorBoard ──────────────────────────────────────────────────────
        writer.add_scalar("Loss/train", tr_loss,  epoch)
        writer.add_scalar("Loss/val",   val_loss, epoch)
        writer.add_scalar("Confianza/c_t_val", val_conf["c_t"], epoch)
        writer.add_scalar("Confianza/c_a_val", val_conf["c_a"], epoch)
        writer.add_scalar("Confianza/c_v_val", val_conf["c_v"], epoch)
        for k, v in val_met.items():
            writer.add_scalar(f"Metricas/{k}", v, epoch)

        # ── Stats cada 5 épocas ──────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == args.epochs:
            print_model_stats(model, epoch, log)
            log("")
            log(f"  Confiabilidad media en validación:")
            log.kv("    c_t (Texto)", f"{val_conf['c_t']:.4f}")
            log.kv("    c_a (Audio)", f"{val_conf['c_a']:.4f}")
            log.kv("    c_v (Video)", f"{val_conf['c_v']:.4f}")
            log.kv("    LR actual",   f"{scheduler.get_last_lr()[0]:.2e}")
            log("")

        # Historial
        history.append({
            "epoch":    epoch,
            "tr_loss":  round(tr_loss,  6),
            "val_loss": round(val_loss, 6),
            "val_metrics": {k: round(v, 6) for k, v in val_met.items()},
            "val_conf": {k: round(v, 6) for k, v in val_conf.items()},
            "elapsed_s": round(elapsed, 2),
            "is_best":  bool(is_best),
        })

    total_train_time = time.time() - total_start

    # ── Evaluación final en test ──────────────────────────────────────────────
    log.section("Evaluación en TEST (mejor checkpoint)")
    log(f"  Cargando checkpoint de época {best_epoch} ...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_loss, test_met, test_conf = eval_epoch(
        model, loaders["test"], loss_fn, device, args.task, "test"
    )

    log("")
    log.kv("  Pérdida test", f"{test_loss:.4f}")
    log("")
    log("  Métricas test:")
    for k, v in test_met.items():
        log.kv(f"    {k}", f"{v:.4f}")
    log("")
    log("  Confiabilidad media en test:")
    log.kv("    c_t (Texto)", f"{test_conf['c_t']:.4f}")
    log.kv("    c_a (Audio)", f"{test_conf['c_a']:.4f}")
    log.kv("    c_v (Video)", f"{test_conf['c_v']:.4f}")

    # ── Resumen final y archivos guardados ───────────────────────────────────
    log.section("Resumen de la ejecución")

    log.kv("Épocas completadas",     args.epochs)
    log.kv("Mejor época",            best_epoch)
    log.kv("Mejor val_loss",         f"{best_val_loss:.4f}")
    log.kv("Tiempo total",           f"{total_train_time/60:.1f} min")
    log.kv("Tiempo promedio/época",  f"{total_train_time/args.epochs:.1f}s")
    log("")

    log("  Archivos generados:")
    log.kv("    Mejor checkpoint",  str(best_ckpt))
    log.kv("    Último checkpoint", str(last_ckpt))

    # Guardar métricas en JSON
    metrics_path = exp_dir / "metrics.json"
    result = {
        "config":       vars(args),
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss":    test_loss,
        "test_metrics": test_met,
        "test_conf":    test_conf,
        "total_train_time_s": round(total_train_time, 1),
        "history":      history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.kv("    Métricas JSON",     str(metrics_path))
    log.kv("    Log completo",      str(exp_dir / "run.log"))
    log.kv("    TensorBoard",       str(exp_dir / "tensorboard"))

    log("")
    log("  Contenido del checkpoint (best_model.pt):")
    log("    - model state_dict  (pesos del modelo)")
    log("    - optimizer state_dict")
    log("    - epoch, val_loss, val_metrics")
    log("    - args (config completa)")
    log("")
    log("  Para cargar el modelo entrenado:")
    log("    import torch")
    log("    from model import AMFuse")
    log(f"    ckpt  = torch.load('{best_ckpt}')")
    log(f"    model = AMFuse(task='{args.task}', d={args.d_hidden})")
    log("    model.load_state_dict(ckpt['model'])")
    log("    model.eval()")
    log("")
    log("  Para ver las curvas de entrenamiento:")
    log(f"    tensorboard --logdir \"{exp_dir / 'tensorboard'}\"")

    log.section("Fin del entrenamiento")
    log.timestamp("Fecha/hora de cierre:")

    # TensorBoard final
    writer.add_hparams(
        {k: str(v) for k, v in vars(args).items()},
        {"hparam/test_loss": test_loss,
         **{f"hparam/{k}": v for k, v in test_met.items()}}
    )
    writer.close()
    log.close()


if __name__ == "__main__":
    main()
