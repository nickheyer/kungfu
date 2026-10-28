use crate::basic::Linear;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// LFM2 short convolution mixer, in projection to B C x, causal depthwise conv over B times x, C gate
pub struct ShortConv {
    pub n_embd: usize,
    pub l_cache: usize,
    pub in_proj: Linear,
    pub conv: Wid,
    pub out_proj: Linear,
}

impl Layer for ShortConv {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        shortconv_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.in_proj.weights();
        w.push(self.conv);
        w.extend(self.out_proj.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::Recurrent { conv_state: (self.l_cache - 1) * self.n_embd, ssm_state: 0 }]
    }
}

use kf_core::Op;

/// In projection split into b, c and x, the gated causal conv over b times x, then c times the result
pub fn shortconv_graph(g: &mut GraphBuilder, s: &ShortConv, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let e = s.n_embd;
    let bcx = s.in_proj.apply(g, x);
    let b = g.cols(bcx, 0, e);
    let c = g.cols(bcx, e, e);
    let xx = g.cols(bcx, 2 * e, e);
    let bx = g.mul(b, xx);
    let conv = g.push(Op::CausalConv { layer: ctx.layer, slot: 0, kernel: s.l_cache, dilation: 1, weight: s.conv, bias: None }, &[bx]);
    let y = g.mul(c, conv);
    Ok(s.out_proj.apply(g, y))
}
