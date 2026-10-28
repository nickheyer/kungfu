use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, Op, StateNeed, Tid, Wid};

pub struct Identity;

impl Layer for Identity {
    fn forward(&self, _g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        Ok(x)
    }
    fn weights(&self) -> Vec<Wid> {
        Vec::new()
    }
}

/// Token table lookup, scaled, with an optional learned position table added and id substitution first
pub struct Embedding {
    pub w: Wid,
    pub scale: f32,
    pub pos: Option<Wid>,
    pub subst: Option<std::sync::Arc<Vec<(u32, f32)>>>,
}

impl Embedding {
    pub fn plain(w: Wid) -> Self {
        Self { w, scale: 1.0, pos: None, subst: None }
    }
}

impl Layer for Embedding {
    fn forward(&self, g: &mut GraphBuilder, _x: Tid, ctx: &Ctx) -> Result<Tid> {
        let ids = match &self.subst {
            Some(t) => g.push(Op::MapTokens { table: t.clone(), default: None }, &[ctx.tokens]),
            None => ctx.tokens,
        };
        let e = g.push(Op::Embed { weight: self.w }, &[ids]);
        let e = g.scale(e, self.scale);
        Ok(match self.pos {
            Some(p) => {
                let pe = g.push(Op::Gather { weight: p }, &[ctx.positions]);
                g.add(e, pe)
            }
            None => e,
        })
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.w];
        w.extend(self.pos);
        w
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub w: Wid,
    pub b: Option<Wid>,
    pub s: Option<Wid>,
}

impl Linear {
    pub fn new(w: Wid) -> Self {
        Self { w, b: None, s: None }
    }
    pub fn apply(&self, g: &mut GraphBuilder, x: Tid) -> Tid {
        g.push(Op::MatMul { weight: self.w, bias: self.b, scale: self.s }, &[x])
    }
}

impl Layer for Linear {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        Ok(self.apply(g, x))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.w];
        w.extend(self.b);
        w.extend(self.s);
        w
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RmsNorm {
    pub w: Option<Wid>,
    pub b: Option<Wid>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn weightless(eps: f32) -> Self {
        Self { w: None, b: None, eps }
    }
    pub fn apply(&self, g: &mut GraphBuilder, x: Tid) -> Tid {
        g.push(Op::RmsNorm { weight: self.w, bias: self.b, eps: self.eps }, &[x])
    }
    /// Normalizes each head's slice of a [tokens, heads * dim] tensor with one shared weight
    pub fn apply_per_head(&self, g: &mut GraphBuilder, x: Tid, n_tokens: usize, n_head: usize, dim: usize) -> Tid {
        let r = g.reshape(x, n_tokens * n_head, dim);
        let n = self.apply(g, r);
        g.reshape(n, n_tokens, n_head * dim)
    }
}

impl Layer for RmsNorm {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        Ok(self.apply(g, x))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w: Vec<Wid> = self.w.into_iter().collect();
        w.extend(self.b);
        w
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LayerNorm {
    pub w: Option<Wid>,
    pub b: Option<Wid>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn apply(&self, g: &mut GraphBuilder, x: Tid) -> Tid {
        g.push(Op::LayerNorm { weight: self.w, bias: self.b, eps: self.eps }, &[x])
    }
}

impl Layer for LayerNorm {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        Ok(self.apply(g, x))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w: Vec<Wid> = self.w.into_iter().collect();
        w.extend(self.b);
        w
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Layer,
}

/// How a head norm weight is laid out, one row for all heads, a row per head, the whole width, or each half head
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadLayout {
    Shared,
    PerHead,
    FullWidth,
    HalfHead,
}

/// Norm over the heads of a [tokens, heads * dim] tensor
#[derive(Clone, Copy, Debug)]
pub struct HeadNorm {
    pub kind: NormKind,
    pub w: Option<Wid>,
    pub b: Option<Wid>,
    pub layout: HeadLayout,
    pub eps: f32,
}

impl HeadNorm {
    pub fn rms_shared(w: Wid, eps: f32) -> Self {
        Self { kind: NormKind::Rms, w: Some(w), b: None, layout: HeadLayout::Shared, eps }
    }
    pub fn rms_weightless(eps: f32) -> Self {
        Self { kind: NormKind::Rms, w: None, b: None, layout: HeadLayout::Shared, eps }
    }
    fn bare(&self, g: &mut GraphBuilder, x: Tid, with_affine: bool) -> Tid {
        let (w, b) = if with_affine { (self.w, self.b) } else { (None, None) };
        match self.kind {
            NormKind::Rms => g.push(Op::RmsNorm { weight: w, bias: b, eps: self.eps }, &[x]),
            NormKind::Layer => g.push(Op::LayerNorm { weight: w, bias: b, eps: self.eps }, &[x]),
        }
    }
    pub fn apply(&self, g: &mut GraphBuilder, x: Tid, n_tokens: usize, n_head: usize, dim: usize) -> Tid {
        match self.layout {
            HeadLayout::Shared => {
                let r = g.reshape(x, n_tokens * n_head, dim);
                let n = self.bare(g, r, true);
                g.reshape(n, n_tokens, n_head * dim)
            }
            HeadLayout::FullWidth => self.bare(g, x, true),
            HeadLayout::PerHead => {
                let r = g.reshape(x, n_tokens * n_head, dim);
                let n = self.bare(g, r, false);
                let mut n = g.reshape(n, n_tokens, n_head * dim);
                if let Some(w) = self.w {
                    n = g.push(Op::MulWeight { weight: w }, &[n]);
                }
                if let Some(b) = self.b {
                    n = g.push(Op::AddWeight { weight: b }, &[n]);
                }
                n
            }
            HeadLayout::HalfHead => {
                let r = g.reshape(x, n_tokens * n_head * 2, dim / 2);
                let n = self.bare(g, r, false);
                let mut n = g.reshape(n, n_tokens * n_head, dim);
                if let Some(w) = self.w {
                    n = g.push(Op::MulWeight { weight: w }, &[n]);
                }
                g.reshape(n, n_tokens, n_head * dim)
            }
        }
    }
    pub fn weights(&self) -> Vec<Wid> {
        let mut w: Vec<Wid> = self.w.into_iter().collect();
        w.extend(self.b);
        w
    }
}

/// x_scale times x plus scale times inner applied to a normed x, with an optional norm on the way back
pub struct Residual {
    pub norm: Box<dyn Layer>,
    pub inner: Box<dyn Layer>,
    pub post: Option<Box<dyn Layer>>,
    pub scale: f32,
    pub x_scale: f32,
}

impl Residual {
    pub fn new(norm: impl Layer + 'static, inner: impl Layer + 'static) -> Self {
        Self { norm: Box::new(norm), inner: Box::new(inner), post: None, scale: 1.0, x_scale: 1.0 }
    }
    pub fn with_post(mut self, post: impl Layer + 'static) -> Self {
        self.post = Some(Box::new(post));
        self
    }
    pub fn scaled(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
    pub fn x_scaled(mut self, x_scale: f32) -> Self {
        self.x_scale = x_scale;
        self
    }
}

impl Layer for Residual {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let mut inner_ctx = ctx.clone();
        inner_ctx.residual_input = x;
        let h = self.norm.forward(g, x, &inner_ctx)?;
        let mut y = self.inner.forward(g, h, &inner_ctx)?;
        if let Some(post) = &self.post {
            y = post.forward(g, y, &inner_ctx)?;
        }
        let y = g.scale(y, self.scale);
        let xs = g.scale(x, self.x_scale);
        Ok(g.add(xs, y))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.norm.weights();
        w.extend(self.inner.weights());
        if let Some(p) = &self.post {
            w.extend(p.weights());
        }
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        self.inner.states()
    }
}

/// alpha times x plus beta times the branch, the norm preserving residual of Muse Spark
pub struct ScaledResidual {
    pub norm: Box<dyn Layer>,
    pub inner: Box<dyn Layer>,
    pub post: Option<Box<dyn Layer>>,
    pub alpha: Wid,
    pub beta: Wid,
}

impl Layer for ScaledResidual {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let mut inner_ctx = ctx.clone();
        inner_ctx.residual_input = x;
        let h = self.norm.forward(g, x, &inner_ctx)?;
        let mut y = self.inner.forward(g, h, &inner_ctx)?;
        if let Some(post) = &self.post {
            y = post.forward(g, y, &inner_ctx)?;
        }
        let xs = g.push(Op::MulWeight { weight: self.alpha }, &[x]);
        let ys = g.push(Op::MulWeight { weight: self.beta }, &[y]);
        Ok(g.add(xs, ys))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.norm.weights();
        w.extend(self.inner.weights());
        if let Some(p) = &self.post {
            w.extend(p.weights());
        }
        w.extend([self.alpha, self.beta]);
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        self.inner.states()
    }
}

pub struct Branch {
    pub norm: Option<Box<dyn Layer>>,
    pub layer: Box<dyn Layer>,
    pub scale: f32,
}

/// Scaled sum of branches applied to one input, each after its own optional norm, then an optional post layer
pub struct Parallel {
    pub norm: Option<Box<dyn Layer>>,
    pub branches: Vec<Branch>,
    pub post: Option<Box<dyn Layer>>,
}

impl Layer for Parallel {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let base = match &self.norm {
            Some(n) => n.forward(g, x, ctx)?,
            None => x,
        };
        let mut acc: Option<Tid> = None;
        for b in &self.branches {
            let h = match &b.norm {
                Some(n) => n.forward(g, base, ctx)?,
                None => base,
            };
            let y = b.layer.forward(g, h, ctx)?;
            let y = g.scale(y, b.scale);
            acc = Some(match acc {
                Some(a) => g.add(a, y),
                None => y,
            });
        }
        let mut out = acc.ok_or_else(|| anyhow::anyhow!("parallel layer with no branches"))?;
        if let Some(p) = &self.post {
            out = p.forward(g, out, ctx)?;
        }
        Ok(out)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w: Vec<Wid> = self.norm.iter().flat_map(|n| n.weights()).collect();
        for b in &self.branches {
            w.extend(b.norm.iter().flat_map(|n| n.weights()));
            w.extend(b.layer.weights());
        }
        w.extend(self.post.iter().flat_map(|p| p.weights()));
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        self.branches.iter().flat_map(|b| b.layer.states()).collect()
    }
}

/// y is x plus attention, out is y plus a dense branch of y plus a second branch of the block input
pub struct SplitResidual {
    pub attn_norm: Box<dyn Layer>,
    pub attn: Box<dyn Layer>,
    pub dense_norm: Box<dyn Layer>,
    pub dense: Box<dyn Layer>,
    pub side_norm: Box<dyn Layer>,
    pub side: Box<dyn Layer>,
}

impl Layer for SplitResidual {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let mut c = ctx.clone();
        c.residual_input = x;
        let h = self.attn_norm.forward(g, x, &c)?;
        let a = self.attn.forward(g, h, &c)?;
        let y = g.add(x, a);
        c.residual_input = y;
        let hd = self.dense_norm.forward(g, y, &c)?;
        let d = self.dense.forward(g, hd, &c)?;
        c.residual_input = x;
        let hs = self.side_norm.forward(g, x, &c)?;
        let s = self.side.forward(g, hs, &c)?;
        let yd = g.add(y, d);
        Ok(g.add(yd, s))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.attn_norm.weights();
        w.extend(self.attn.weights());
        w.extend(self.dense_norm.weights());
        w.extend(self.dense.weights());
        w.extend(self.side_norm.weights());
        w.extend(self.side.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        let mut s = self.attn.states();
        s.extend(self.dense.states());
        s.extend(self.side.states());
        s
    }
}

/// Multiplies the whole stream by a factor and an optional weight of one or every element
pub struct StreamScale {
    pub weight: Option<Wid>,
    pub factor: f32,
}

impl Layer for StreamScale {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        let mut y = g.scale(x, self.factor);
        if let Some(w) = self.weight {
            y = g.push(Op::MulWeight { weight: w }, &[y]);
        }
        Ok(y)
    }
    fn weights(&self) -> Vec<Wid> {
        self.weight.into_iter().collect()
    }
}

/// Adds the normed token embedding times a scalar weight back into the stream
pub struct EmbedSkip {
    pub eps: f32,
    pub scale: Wid,
}

impl Layer for EmbedSkip {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let e = g.push(Op::RmsNorm { weight: None, bias: None, eps: self.eps }, &[ctx.embed]);
        let e = g.push(Op::MulWeight { weight: self.scale }, &[e]);
        Ok(g.add(x, e))
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.scale]
    }
}

/// Applies a layer to one slice of the columns and leaves the rest untouched
pub struct OnCols {
    pub start: usize,
    pub len: usize,
    pub total: usize,
    pub inner: Box<dyn Layer>,
}

impl Layer for OnCols {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let mid = g.cols(x, self.start, self.len);
        let mid = self.inner.forward(g, mid, ctx)?;
        let mut out = mid;
        if self.start > 0 {
            let head = g.cols(x, 0, self.start);
            out = g.concat(head, out);
        }
        if self.start + self.len < self.total {
            let tail = g.cols(x, self.start + self.len, self.total - self.start - self.len);
            out = g.concat(out, tail);
        }
        Ok(out)
    }
    fn weights(&self) -> Vec<Wid> {
        self.inner.weights()
    }
    fn states(&self) -> Vec<StateNeed> {
        self.inner.states()
    }
}

/// Multiplies by a weight and then rms normalizes without a weight, the post norm order of Muse Spark
pub struct MulRms {
    pub w: Wid,
    pub eps: f32,
}

impl Layer for MulRms {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        let m = g.push(Op::MulWeight { weight: self.w }, &[x]);
        Ok(g.push(Op::RmsNorm { weight: None, bias: None, eps: self.eps }, &[m]))
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.w]
    }
}
