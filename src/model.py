"""
AMFuse: Adaptive Multimodal Fusion with Missing-Modality Awareness.

Pipeline (6 módulos, según Capítulo 3 de la tesis):
  M1  Codificador de estado de modalidades  – máscara binaria m ∈ {0,1}^3
  M2  Codificadores unimodales              – proyectar features -> R^d (d=256)
  M3A Generador de modalidades faltantes    – CrossAttn para sintetizar ausentes
  M3B Estimador de confiabilidad            – escalar c_i ∈ [0,1] por modalidad (Ec. 3.4)
  M4  Atención cruzada con ancla dinámica   – selección de ancla + cross-attn (Ec. 3.5-3.7)
  M5  Fusión dinámica consciente de conf.   – pesos DAF × confiabilidad (Ec. 3.8-3.11)
  M6  Clasificador                          – MLP 2 capas -> regresión o 3 clases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Hiperparámetros ──────────────────────────────────────────────────────────
D_TEXT  = 768
D_AUDIO = 74
D_VIDEO = 35
D_HIDDEN  = 256
N_HEADS   = 8
BETA_INIT = 0.5   # penalización para modalidades generadas (Ec. 3.4)

_MODAL_NAMES = {0: "Texto", 1: "Audio", 2: "Video"}


# ── Módulo 2: Proyectores unimodales ─────────────────────────────────────────

class UnimodalProjectors(nn.Module):
    """
    Proyección lineal desde el espacio de features brutos -> R^{D_HIDDEN}.
    Los encoders pre-entrenados (BERT/COVAREP/FACET) corren fuera del modelo;
    sus outputs son los inputs aquí.
    """

    def __init__(self, d_text=D_TEXT, d_audio=D_AUDIO, d_video=D_VIDEO, d=D_HIDDEN):
        super().__init__()
        self.proj_t = nn.Linear(d_text,  d)
        self.proj_a = nn.Linear(d_audio, d)
        self.proj_v = nn.Linear(d_video, d)

    def forward(self, text, audio, video, mt, ma, mv, verbose_fn=None):
        h_t = self.proj_t(text)
        h_a = self.proj_a(audio)
        h_v = self.proj_v(video)

        # Zerear modalidades ausentes (su proyección no tiene sentido)
        h_t = h_t * mt.float().unsqueeze(1)
        h_a = h_a * ma.float().unsqueeze(1)
        h_v = h_v * mv.float().unsqueeze(1)

        # Guardar contra Inf/NaN de features brutos (p.ej. COVAREP frames no vocalidos)
        h_t = torch.nan_to_num(h_t, nan=0.0, posinf=0.0, neginf=0.0)
        h_a = torch.nan_to_num(h_a, nan=0.0, posinf=0.0, neginf=0.0)
        h_v = torch.nan_to_num(h_v, nan=0.0, posinf=0.0, neginf=0.0)

        if verbose_fn:
            B = text.size(0)
            n_t, n_a, n_v = mt.sum().item(), ma.sum().item(), mv.sum().item()
            verbose_fn(
                f"    [M2] Proyección unimodal completada — "
                f"Texto: {n_t}/{B} presentes  "
                f"Audio: {n_a}/{B} presentes  "
                f"Video: {n_v}/{B} presentes"
            )
            verbose_fn(
                f"    [M2] Normas embeddings -> "
                f"h_t={h_t.norm(dim=1).mean():.3f}  "
                f"h_a={h_a.norm(dim=1).mean():.3f}  "
                f"h_v={h_v.norm(dim=1).mean():.3f}"
            )

        return h_t, h_a, h_v


# ── Módulo 3A: Generador de modalidades faltantes ────────────────────────────

class ModalityGenerator(nn.Module):
    """
    Mini-transformer que genera el embedding sintético de UNA modalidad faltante
    a partir de las dos disponibles (Ec. 3.3).
    Una instancia por modalidad potencialmente faltante.
    """

    def __init__(self, d=D_HIDDEN, n_heads=N_HEADS, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d)
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, query_src, key_val_src):
        q  = query_src.unsqueeze(1)
        kv = key_val_src.unsqueeze(1)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.drop(out)
        out = self.norm(out.squeeze(1) + query_src)
        out = self.norm2(out + self.ff(out))
        return out


class MissingModalityGenerator(nn.Module):
    """
    Módulo 3A: tres generadores independientes, uno por modalidad.
    Genera embeddings sintéticos para todas las modalidades ausentes en el batch.
    Actualiza h y tau (tipo: 1.0=real, 0.0=generado).
    """

    def __init__(self, d=D_HIDDEN, n_heads=N_HEADS, dropout=0.0):
        super().__init__()
        self.gen_t = ModalityGenerator(d, n_heads, dropout)
        self.gen_a = ModalityGenerator(d, n_heads, dropout)
        self.gen_v = ModalityGenerator(d, n_heads, dropout)

    def _best_pair(self, h1, h2, m1, m2):
        query   = h1
        key_val = torch.where(m2.unsqueeze(1).bool(), h2, h1)
        return query, key_val

    def forward(self, h_t, h_a, h_v, mt, ma, mv, verbose_fn=None):
        tau_t = mt.float()
        tau_a = ma.float()
        tau_v = mv.float()

        # Generar texto faltante desde audio+video
        need_t = (mt == 0)
        if need_t.any():
            q, kv = self._best_pair(h_a, h_v, ma, mv)
            syn = self.gen_t(q, kv)
            h_t  = torch.where(need_t.unsqueeze(1), syn, h_t)
            tau_t = torch.where(need_t, torch.zeros_like(tau_t), tau_t)
            if verbose_fn:
                verbose_fn(f"    [M3A] Generando TEXTO para {need_t.sum().item()} muestras (audio+video como referencia)")

        # Generar audio faltante desde texto+video
        need_a = (ma == 0)
        if need_a.any():
            q, kv = self._best_pair(h_t, h_v, mt, mv)
            syn = self.gen_a(q, kv)
            h_a  = torch.where(need_a.unsqueeze(1), syn, h_a)
            tau_a = torch.where(need_a, torch.zeros_like(tau_a), tau_a)
            if verbose_fn:
                verbose_fn(f"    [M3A] Generando AUDIO para {need_a.sum().item()} muestras (texto+video como referencia)")

        # Generar video faltante desde texto+audio
        need_v = (mv == 0)
        if need_v.any():
            q, kv = self._best_pair(h_t, h_a, mt, ma)
            syn = self.gen_v(q, kv)
            h_v  = torch.where(need_v.unsqueeze(1), syn, h_v)
            tau_v = torch.where(need_v, torch.zeros_like(tau_v), tau_v)
            if verbose_fn:
                verbose_fn(f"    [M3A] Generando VIDEO para {need_v.sum().item()} muestras (texto+audio como referencia)")

        if verbose_fn and not need_t.any() and not need_a.any() and not need_v.any():
            verbose_fn("    [M3A] Sin modalidades faltantes — sin generación necesaria")

        return h_t, h_a, h_v, tau_t, tau_a, tau_v

    def compute_recon_loss(self, h_t, h_a, h_v, mt, ma, mv):
        """
        Loss de reconstrucción auto-supervisado: para cada modalidad presente,
        genera su embedding desde las otras dos y compara con el real.
        Solo cuenta muestras donde haya al menos una modalidad de contexto válida.
        """
        loss  = h_t.new_zeros(())
        count = 0

        # Reconstruir TEXTO desde audio+video (solo donde texto está presente y hay contexto)
        has_t = (mt == 1) & ((ma == 1) | (mv == 1))
        if has_t.any():
            q, kv = self._best_pair(h_a[has_t], h_v[has_t], ma[has_t], mv[has_t])
            loss  = loss + F.mse_loss(self.gen_t(q, kv), h_t[has_t].detach())
            count += 1

        # Reconstruir AUDIO desde texto+video
        has_a = (ma == 1) & ((mt == 1) | (mv == 1))
        if has_a.any():
            q, kv = self._best_pair(h_t[has_a], h_v[has_a], mt[has_a], mv[has_a])
            loss  = loss + F.mse_loss(self.gen_a(q, kv), h_a[has_a].detach())
            count += 1

        # Reconstruir VIDEO desde texto+audio
        has_v = (mv == 1) & ((mt == 1) | (ma == 1))
        if has_v.any():
            q, kv = self._best_pair(h_t[has_v], h_a[has_v], mt[has_v], ma[has_v])
            loss  = loss + F.mse_loss(self.gen_v(q, kv), h_v[has_v].detach())
            count += 1

        return loss / max(count, 1)


# ── Módulo 3B: Estimador de confiabilidad ────────────────────────────────────

class ConfidenceEstimator(nn.Module):
    """
    Módulo 3B (contribución original): MLP 2 capas -> sigmoid -> escalado por beta
    para modalidades generadas (Ec. 3.4).

    beta es ahora un nn.Parameter learnable (beta = sigmoid(_beta_logit)).
    Inicializado en sigmoid(0) = 0.5, igual que BETA_INIT, pero el optimizador
    puede ajustarlo durante el entrenamiento para aprender el penalty óptimo.
    """

    def __init__(self, d=D_HIDDEN):
        super().__init__()
        self.phi_t = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        self.phi_a = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        self.phi_v = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        # sigmoid(0) = 0.5 = BETA_INIT; learnable, siempre en (0, 1)
        self._beta_logit = nn.Parameter(torch.zeros(1))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self._beta_logit)

    def forward(self, h_t, h_a, h_v, tau_t, tau_a, tau_v, verbose_fn=None):
        beta = self.beta   # evaluar una vez por forward

        def _conf(phi, h, tau):
            raw   = torch.sigmoid(phi(h).squeeze(-1))
            scale = tau + (1.0 - tau) * beta
            return raw * scale

        c_t = _conf(self.phi_t, h_t, tau_t)
        c_a = _conf(self.phi_a, h_a, tau_a)
        c_v = _conf(self.phi_v, h_v, tau_v)

        if verbose_fn:
            verbose_fn(
                f"    [M3B] Confiabilidad estimada -> "
                f"c_t={c_t.mean():.3f}±{c_t.std():.3f}  "
                f"c_a={c_a.mean():.3f}±{c_a.std():.3f}  "
                f"c_v={c_v.mean():.3f}±{c_v.std():.3f}  "
                f"(beta={beta.item():.3f})"
            )

        return c_t, c_a, c_v


# ── Módulo 4: Atención cruzada con ancla dinámica ────────────────────────────

class DynamicAnchorCrossAttn(nn.Module):
    """
    Módulo 4: mezcla suave ponderada por confianza (Ec. 3.5-3.7, versión diferenciable).

    En lugar de un argmax duro, los pesos w = softmax(c_eff / T) mezclan todas las
    cross-atenciones posibles. Cuando una modalidad domina (w_i → 1) el comportamiento
    converge al ancla dura original. La temperatura T es learnable (init=1).

    Interpretación de h_t_out:
      w_t * h_t       — texto actúa como ancla (se queda sin cambio)
      w_a * e_{a→t}   — audio es ancla: texto es cross-atendido desde audio
      w_v * e_{v→t}   — video es ancla: texto es cross-atendido desde video
    Análogo para h_a_out y h_v_out.

    Ventaja frente al argmax: ∂loss/∂c_i fluye a través de w directamente hacia M3B,
    resolviendo el cortocircuito de gradiente que causaba el colapso de c_a y c_v.
    """

    def __init__(self, d=D_HIDDEN, n_heads=N_HEADS):
        super().__init__()
        self.attn_t_anchor = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.attn_a_anchor = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.attn_v_anchor = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.temperature   = nn.Parameter(torch.ones(1))   # T learnable, init=1

    def _cross(self, attn_module, anchor_h, other_h):
        q  = anchor_h.unsqueeze(1)
        kv = other_h.unsqueeze(1)
        out, _ = attn_module(q, kv, kv)
        return out.squeeze(1)

    def forward(self, h_t, h_a, h_v, c_t, c_a, c_v, mt, ma, mv, verbose_fn=None):
        eff_t = c_t * mt.float()
        eff_a = c_a * ma.float()
        eff_v = c_v * mv.float()

        confs = torch.stack([eff_t, eff_a, eff_v], dim=1)          # [B, 3]
        temp  = self.temperature.clamp(min=0.1)                     # evitar T→0 inestable
        w     = F.softmax(confs / temp, dim=1)                      # [B, 3], diferenciable

        if verbose_fn:
            anchor_idx = confs.argmax(dim=1)   # solo para diagnóstico
            cnt = [(anchor_idx == i).sum().item() for i in range(3)]
            verbose_fn(
                f"    [M4]  Ancla dominante (soft) -> "
                f"Texto: {cnt[0]}  Audio: {cnt[1]}  Video: {cnt[2]}  muestras  "
                f"(T={temp.item():.3f})"
            )

        # Todas las combinaciones de cross-atención (6 en total)
        e_t_a = self._cross(self.attn_t_anchor, h_t, h_a)  # texto como Q, audio como KV
        e_t_v = self._cross(self.attn_t_anchor, h_t, h_v)
        e_a_t = self._cross(self.attn_a_anchor, h_a, h_t)
        e_a_v = self._cross(self.attn_a_anchor, h_a, h_v)
        e_v_t = self._cross(self.attn_v_anchor, h_v, h_t)
        e_v_a = self._cross(self.attn_v_anchor, h_v, h_a)

        # Mezcla suave: cada modalidad recibe su embedding "de ancla" ponderado por w
        h_t_out = w[:, 0:1] * h_t   + w[:, 1:2] * e_a_t + w[:, 2:3] * e_v_t
        h_a_out = w[:, 0:1] * e_t_a + w[:, 1:2] * h_a   + w[:, 2:3] * e_v_a
        h_v_out = w[:, 0:1] * e_t_v + w[:, 1:2] * e_a_v + w[:, 2:3] * h_v

        return h_t_out, h_a_out, h_v_out


# ── Módulo 5: Fusión dinámica consciente de confiabilidad ────────────────────

class ConfidenceAwareFusion(nn.Module):
    """
    Módulo 5: pesos base DAF (Ec. 3.8) modulados por confiabilidad (Ec. 3.9),
    renormalizados (Ec. 3.10), suma ponderada final (Ec. 3.11).
    """

    def __init__(self, d=D_HIDDEN):
        super().__init__()
        self.weight_mlp = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

    def forward(self, h_t, h_a, h_v, c_t, c_a, c_v, verbose_fn=None):
        cat      = torch.cat([h_t, h_a, h_v], dim=-1)
        lam      = F.softmax(self.weight_mlp(cat), dim=-1)         # [B, 3]
        c        = torch.stack([c_t, c_a, c_v], dim=1)
        lam_p    = lam * c
        lam_pp   = lam_p / (lam_p.sum(dim=1, keepdim=True) + 1e-8)

        if verbose_fn:
            verbose_fn(
                f"    [M5]  Pesos base λ     -> "
                f"t={lam[:,0].mean():.3f}  a={lam[:,1].mean():.3f}  v={lam[:,2].mean():.3f}"
            )
            verbose_fn(
                f"    [M5]  Pesos finales λ'' -> "
                f"t={lam_pp[:,0].mean():.3f}  a={lam_pp[:,1].mean():.3f}  v={lam_pp[:,2].mean():.3f}"
            )

        z = (lam_pp[:, 0:1] * h_t
           + lam_pp[:, 1:2] * h_a
           + lam_pp[:, 2:3] * h_v)
        return z


# ── Módulo 6: Clasificador ───────────────────────────────────────────────────

class Classifier(nn.Module):
    """Módulo 6: MLP 2 capas con ReLU. Ec. 3.12."""

    def __init__(self, d=D_HIDDEN, n_classes: int = 1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, n_classes),
        )

    def forward(self, z):
        return self.net(z)


# ── AMFuse completo ──────────────────────────────────────────────────────────

class AMFuse(nn.Module):
    """
    Pipeline completo de AMFuse.

    Args:
        task    : 'regression' (score MOSEI) o 'classification' (3 clases)
        d       : dimensión latente compartida (default 256)
        verbose : si True, imprime detalle de cada módulo en el primer batch
    """

    def __init__(self, task: str = "regression", d: int = D_HIDDEN, dropout: float = 0.0):
        super().__init__()
        assert task in ("regression", "classification")
        self.task = task

        self.projectors = UnimodalProjectors(d=d)              # M2
        self.generator  = MissingModalityGenerator(d=d, dropout=dropout)  # M3A
        self.confidence = ConfidenceEstimator(d=d)             # M3B
        self.cross_attn = DynamicAnchorCrossAttn(d=d)          # M4
        self.fusion     = ConfidenceAwareFusion(d=d)           # M5
        n_out           = 1 if task == "regression" else 3
        self.classifier = Classifier(d=d, n_classes=n_out, dropout=dropout)  # M6

        # Controla el detalle de prints durante el forward
        # Asignar una función callable para activar; None = silencioso
        self._verbose_fn = None

    def set_verbose(self, fn=print):
        """Activa modo verbose. fn es el callable que recibe los strings."""
        self._verbose_fn = fn

    def set_silent(self):
        """Desactiva modo verbose."""
        self._verbose_fn = None

    def forward(self, text, audio, video, mask_t, mask_a, mask_v):
        vfn = self._verbose_fn

        if vfn:
            B   = text.size(0)
            n_t = mask_t.sum().item()
            n_a = mask_a.sum().item()
            n_v = mask_v.sum().item()
            vfn(f"  [M1]  Estado de modalidades — batch={B}  "
                f"Texto={n_t}  Audio={n_a}  Video={n_v}  presentes")

        # M2
        if vfn: vfn("  [M2]  Proyectores unimodales ->")
        h_t, h_a, h_v = self.projectors(
            text, audio, video, mask_t, mask_a, mask_v, verbose_fn=vfn
        )

        # M3A
        if vfn: vfn("  [M3A] Generador de modalidades faltantes ->")
        h_t, h_a, h_v, tau_t, tau_a, tau_v = self.generator(
            h_t, h_a, h_v, mask_t, mask_a, mask_v, verbose_fn=vfn
        )

        # Loss auxiliar de reconstrucción (solo activo en training)
        if self.training:
            recon_loss = self.generator.compute_recon_loss(h_t, h_a, h_v, mask_t, mask_a, mask_v)
        else:
            recon_loss = h_t.new_zeros(())

        # M3B
        if vfn: vfn("  [M3B] Estimador de confiabilidad ->")
        c_t, c_a, c_v = self.confidence(
            h_t, h_a, h_v, tau_t, tau_a, tau_v, verbose_fn=vfn
        )

        # Spread loss: penaliza el colapso de varianza dentro de los embeddings REALES.
        # Se calcula solo sobre muestras con tau_i=1 (modalidad observada) para evitar
        # que el optimizador use el atajo trivial "real→1, generado→0.5" que maximiza
        # varianza sin aprender calidad diferenciada. Con esta restricción, phi_i debe
        # aprender a distinguir qué embeddings reales son más o menos informativos.
        def _real_var(c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            real = c[tau == 1]
            return real.var() if real.numel() > 1 else c.new_zeros(())

        spread_loss = -(_real_var(c_t, tau_t) + _real_var(c_a, tau_a) + _real_var(c_v, tau_v))

        # M4
        if vfn: vfn("  [M4]  Atención cruzada con ancla dinámica ->")
        h_t, h_a, h_v = self.cross_attn(
            h_t, h_a, h_v, c_t, c_a, c_v, mask_t, mask_a, mask_v, verbose_fn=vfn
        )

        # M5
        if vfn: vfn("  [M5]  Fusión dinámica consciente de confiabilidad ->")
        z = self.fusion(h_t, h_a, h_v, c_t, c_a, c_v, verbose_fn=vfn)

        # M6
        if vfn: vfn("  [M6]  Clasificador ->")
        out = self.classifier(z)
        if vfn: vfn(f"    [M6]  Predicciones -> shape={tuple(out.shape)}  "
                    f"min={out.min().item():.3f}  max={out.max().item():.3f}")

        aux = {
            "c_t":         c_t.detach(),
            "c_a":         c_a.detach(),
            "c_v":         c_v.detach(),
            "recon_loss":  recon_loss,
            "spread_loss": spread_loss,
        }
        return out, aux

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def grad_norms(self):
        """Devuelve norma L2 del gradiente por módulo (después de backward)."""
        modules = {
            "M2_projectors": self.projectors,
            "M3A_generator": self.generator,
            "M3B_confidence": self.confidence,
            "M4_cross_attn": self.cross_attn,
            "M5_fusion":     self.fusion,
            "M6_classifier": self.classifier,
        }
        norms = {}
        for name, mod in modules.items():
            grads = [p.grad.norm().item()
                     for p in mod.parameters()
                     if p.grad is not None]
            norms[name] = sum(grads) / len(grads) if grads else 0.0
        return norms
