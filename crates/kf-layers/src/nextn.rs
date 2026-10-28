use crate::basic::{Linear, RmsNorm};
use crate::model::Block;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// Next token prediction head, drafts one token ahead from the trunk's hidden state
pub struct NextN {
    pub eh_proj: Linear,
    pub enorm: RmsNorm,
    pub hnorm: RmsNorm,
    /// The hnorm weight holds one row per stream
    pub hnorm_per_stream: bool,
    /// The draft's own token table or the main one
    pub embed_tokens: Wid,
    pub head: Linear,
    pub head_norm: Box<dyn Layer>,
    pub block: Block,
    /// Residual streams the block runs on and the width of one
    pub n_streams: usize,
    pub n_embd: usize,
    /// Collapses the streams before the head norm
    pub pre_head: Option<Box<dyn Layer>>,
}

impl Layer for NextN {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        nextn_draft_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.eh_proj.weights();
        w.extend(self.enorm.weights());
        w.extend(self.hnorm.weights());
        w.push(self.embed_tokens);
        w.extend(self.head.weights());
        w.extend(self.head_norm.weights());
        w.extend(self.pre_head.iter().flat_map(|h| h.weights()));
        for p in &self.block.parts {
            w.extend(p.weights());
        }
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        self.block.states()
    }
}

use crate::util::{mul_weight, rms};
use kf_core::Op;

/// The draft block, its input the normed next token embedding joined with the normed trunk state
pub fn nextn_draft_graph(g: &mut GraphBuilder, n: &NextN, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let s = n.n_streams.max(1);
    let e = n.n_embd;
    let emb = g.push(Op::Gather { weight: n.embed_tokens }, &[ctx.tokens]);
    let en = n.enorm.apply(g, emb);
    let (en, hn) = if s > 1 {
        let en = g.push(Op::Tile { n: s }, &[en]);
        let r = g.reshape(x, t * s, e);
        let hn = if n.hnorm_per_stream {
            let r = rms(g, r, None, n.hnorm.eps);
            let r = g.reshape(r, t, s * e);
            match n.hnorm.w {
                Some(w) => mul_weight(g, r, w),
                None => r,
            }
        } else {
            let r = n.hnorm.apply(g, r);
            g.reshape(r, t, s * e)
        };
        (en, hn)
    } else {
        (en, n.hnorm.apply(g, x))
    };
    let en = g.reshape(en, t * s, e);
    let hn = g.reshape(hn, t * s, e);
    let cat = g.concat(en, hn);
    let c = n.eh_proj.apply(g, cat);
    let mut c = g.reshape(c, t, s * e);
    let mut ctx2 = ctx.clone();
    ctx2.layer = n.block.layer;
    ctx2.block_input = c;
    ctx2.residual_input = c;
    let saved = g.layer;
    g.layer = Some(n.block.layer);
    for part in &n.block.parts {
        c = part.forward(g, c, &ctx2)?;
    }
    g.layer = saved;
    if let Some(p) = &n.pre_head {
        c = p.forward(g, c, &ctx2)?;
    }
    let h = n.head_norm.forward(g, c, &ctx2)?;
    Ok(n.head.apply(g, h))
}
