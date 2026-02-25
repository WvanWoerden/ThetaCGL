use core::convert::TryFrom;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use rand_core::{CryptoRng, RngCore};

use super::utils64::{addcarry_u64, subborrow_u64, umull};

#[derive(Clone, Copy, Debug)]
pub struct Gf127([u64; 2]);

impl Gf127 {
    // IMPLEMENTATION NOTES
    // ====================
    //
    // Modulus is: q = 2^127 - 1
    // Internally we only ensure that values x < 2^127 which means
    // the element 0 has two valid representations: 0 and 2^127 - 1

    // Element encoding length (in bytes); always 32 bytes.
    pub const ENCODED_LENGTH: usize = 16;
    pub const BIT_LENGTH: usize = 127; // TODO ONLY USED FOR PRINTING

    // Modulus q in base 2^64 (low-to-high order).
    pub const MODULUS: [u64; 2] = [0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF];
    pub const ZERO: Self = Self([0, 0]);
    pub const ONE: Self = Self([1, 0]);
    pub const MINUS_ONE: Self = Self([0xFFFFFFFFFFFFFFFE, 0x7FFFFFFFFFFFFFFF]);

    // TODO: these are extra constants I needed for the extension
    // construction... Maybe these should be shifted to a specialised
    // extension...
    pub const N: usize = 2; // Number of limbs
    pub const TWO: Self = Self([2, 0]);
    pub const THREE: Self = Self([3, 0]);
    pub const FOUR: Self = Self([4, 0]);
    pub const THREE_INV: Self = Self([0x5555555555555555, 0x5555555555555555]);

    /// Get a new instance containing the provided 128-bit integer,
    /// which is implicitly reduced modulo q. All 128-bit values are
    /// accepted. This function is meant for constant (compile-time)
    /// evaluation; at runtime, consider using `from_w64le()`, which
    /// is faster.
    /// This function expects the two 64-bit limbs in little-endian order.
    pub const fn w64le(x0: u64, x1: u64) -> Self {
        Self(Self::const_mred(x0, x1))
    }

    /// Get a new instance containing the provided 128-bit integer,
    /// which is implicitly reduced modulo q. All 128-bit values are
    /// accepted. This function is meant for constant (compile-time)
    /// evaluation; at runtime, consider using `from_w64be()`, which
    /// is faster.
    /// This function expects the two 64-bit limbs in big-endian order.
    pub const fn w64be(x1: u64, x0: u64) -> Self {
        Self(Self::const_mred(x0, x1))
    }

    /// Get a new instance containing the provided 128-bit integer,
    /// which is implicitly reduced modulo q. All 128-bit values are
    /// accepted.
    #[inline(always)]
    pub fn from_w64le(x0: u64, x1: u64) -> Self {
        let mut r = Self([x0, x1]);
        r.set_reduce_small(0);
        r
    }

    /// Get a new instance containing the provided 256-bit integer,
    /// which is implicitly reduced modulo q. All 256-bit values are
    /// accepted.
    #[inline(always)]
    pub fn from_w64be(x1: u64, x0: u64) -> Self {
        let mut r = Self([x0, x1]);
        r.set_reduce_small(0);
        r
    }

    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        // Raw addition max value
        let (mut d0, mut cc) = addcarry_u64(self.0[0], rhs.0[0], 0);
        let (mut d1, _) = addcarry_u64(self.0[1], rhs.0[1], cc);

        // The max value here is 2*(2^127 - 1) = 2^128 - 2
        // Subtraction of the modulus should happen if the top bit
        // is set.
        // 2^128 - 2 - 2^127 + 1 = 2^127 - 1 is in the allowable
        // range so we only need to perform one reduction

        // Extract top bit
        let m = (d1 >> 63) as u8;

        // Clear top bit (same as subtracting 2^127)
        d1 = d1 & 0x7FFFFFFFFFFFFFFF;

        // Add the top bit back mod p (2^127 = 1 mod p) the result
        // will be smaller than 2^127.
        (d0, cc) = addcarry_u64(d0, 0, m);
        (d1, _) = addcarry_u64(d1, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
    }

    #[inline(always)]
    fn set_sub(&mut self, rhs: &Self) {
        // Raw subtraction.
        let (d0, cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = subborrow_u64(self.0[1], rhs.0[1], cc);

        // If the result is negative, we add q back
        let m = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, Self::MODULUS[0] & m, 0);
        let (d1, _) = addcarry_u64(d1, Self::MODULUS[1] & m, cc);

        self.0[0] = d0;
        self.0[1] = d1;
    }

    // Negate this value (in place).
    #[inline(always)]
    pub fn set_neg(&mut self) {
        // Subtraction from p - self
        // self is in the range [0, 2^127 - 1]
        // and so the result of this subtraction is always in the allowable range?
        let (d0, _) = subborrow_u64(Self::MODULUS[0], self.0[0], 0);
        let (d1, _) = subborrow_u64(Self::MODULUS[1], self.0[1], 0);
        self.0[0] = d0;
        self.0[1] = d1;
    }

    // Conditionally copy the provided value ('a') into self:
    //  - If ctl == 0xFFFFFFFF, then the value of 'a' is copied into self.
    //  - If ctl == 0, then the value of self is unchanged.
    // ctl MUST be equal to 0 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        self.0[0] ^= cw & (self.0[0] ^ a.0[0]);
        self.0[1] ^= cw & (self.0[1] ^ a.0[1]);
    }

    /// Negate this value if ctl is 0xFFFFFFFF; leave it unchanged if
    /// ctl is 0x00000000.
    /// The value of ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn set_condneg(&mut self, ctl: u32) {
        let v = -(self as &Self);
        self.set_cond(&v, ctl);
    }

    // TODO: added this in but maybe we can delete...
    #[inline(always)]
    pub fn set_select(&mut self, a: &Self, b: &Self, ctl: u32) {
        let c = (ctl as u64) | ((ctl as u64) << 32);
        self.0[0] = a.0[0] ^ (c & (a.0[0] ^ b.0[0]));
        self.0[1] = a.0[1] ^ (c & (a.0[1] ^ b.0[1]));
    }

    // Return a value equal to either a0 (if ctl == 0) or a1 (if
    // ctl == 0xFFFFFFFF). Value ctl MUST be either 0 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
        let mut r = *a0;
        r.set_cond(a1, ctl);
        r
    }

    // Conditionally swap two elements: values a and b are exchanged if
    // ctl == 0xFFFFFFFF, or not exchanged if ctl == 0x00000000. Value
    // ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn condswap(a: &mut Self, b: &mut Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        let t = cw & (a.0[0] ^ b.0[0]);
        a.0[0] ^= t;
        b.0[0] ^= t;
        let t = cw & (a.0[1] ^ b.0[1]);
        a.0[1] ^= t;
        b.0[1] ^= t;
    }

    #[inline(always)]
    pub fn set_half(&mut self) {
        // Right-shift the value by 1 bit.
        let d0 = (self.0[0] >> 1) | (self.0[1] << 63);
        let d1 = self.0[1] >> 1;

        // If the dropped bit was 1, then we got (x - 1)/2 and we need
        // to add 1/2 mod q = (q + 1)/2 = 2^126.
        let d1 = d1 + ((self.0[0] & 1).wrapping_neg() & (1u64 << 62));

        // Value is necessarily in range:
        // TODO: double check this...
        self.0[0] = d0;
        self.0[1] = d1;
    }

    #[inline(always)]
    pub fn half(self) -> Self {
        let mut r = self;
        r.set_half();
        r
    }

    // Add hi * 2^128 to this value; this is equivalent to performing an
    // appropriate reduction when the intermediate value has extra bits of
    // value hi.
    // We additionally check if the 127 bit is set and reduce this too, so
    // ultimately we take an integer a + 2^128 * hi and reduce it to ensure
    // it is smaller than 2^127, with a < 2^128.
    #[inline(always)]
    fn set_reduce_small(&mut self, hi: u64) {
        // Add hi*2^128 to the value, which modulo p is the same
        // as adding hi
        let hi = (self.0[1] >> 63) | (hi << 1);
        self.0[1] &= 0x7FFFFFFFFFFFFFFF;

        let (d0, cc) = addcarry_u64(self.0[0], hi, 0);
        let (d1, cc) = addcarry_u64(self.0[1], 0, cc);

        // Now we may have filled the 127th bit again, so perform the reduction
        // once more
        // Extract the top bit and add it to the bottom
        let m = ((cc as u64) << 1) | (d1 >> 63);
        let d1 = d1 & 0x7FFFFFFFFFFFFFFF;
        let (d0, cc) = addcarry_u64(d0, m, 0);
        let (d1, _) = addcarry_u64(d1, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
    }

    // Multiply this value by 2.
    #[inline(always)]
    pub fn set_mul2(&mut self) {
        // Extract the top bit
        let tt = self.0[1] >> 63;

        // Shift left
        self.0[1] = (self.0[1] << 1) | (self.0[0] >> 63);
        self.0[0] = self.0[0] << 1;

        // Wrap around the extracted bit
        self.set_reduce_small(tt);
    }

    #[inline(always)]
    pub fn mul2(self) -> Self {
        let mut r = self;
        r.set_mul2();
        r
    }

    // Multiply this value by 1/3 mod p
    // (not optimised)
    #[inline(always)]
    pub fn set_mul3_inv(&mut self) {
        *self *= &Self::THREE_INV;
    }

    #[inline(always)]
    pub fn mul3_inv(self) -> Self {
        let mut r = self;
        r.set_mul3_inv();
        r
    }

    // Multiply this value by 4.
    #[inline(always)]
    pub fn set_mul4(&mut self) {
        // Extract the top bits
        let tt = self.0[1] >> 62;

        // Shift left
        self.0[1] = (self.0[1] << 2) | (self.0[0] >> 62);
        self.0[0] = self.0[0] << 2;

        // Wrap around the extracted bits
        self.set_reduce_small(tt);
    }

    #[inline(always)]
    pub fn mul4(self) -> Self {
        let mut r = self;
        r.set_mul4();
        r
    }

    // Multiply this value by 8.
    #[inline(always)]
    pub fn set_mul8(&mut self) {
        // Extract the top bits
        let tt = self.0[1] >> 61;

        // Shift left
        self.0[1] = (self.0[1] << 3) | (self.0[0] >> 61);
        self.0[0] = self.0[0] << 3;

        // Wrap around the extracted bits
        self.set_reduce_small(tt);
    }

    #[inline(always)]
    pub fn mul8(self) -> Self {
        let mut r = self;
        r.set_mul8();
        r
    }

    // Multiply this value by 16.
    #[inline(always)]
    pub fn set_mul16(&mut self) {
        // Extract the top bits
        let tt = self.0[1] >> 60;

        // Shift left
        self.0[1] = (self.0[1] << 4) | (self.0[0] >> 60);
        self.0[0] = self.0[0] << 4;

        // Wrap around the extracted bits
        self.set_reduce_small(tt);
    }

    #[inline(always)]
    pub fn mul16(self) -> Self {
        let mut r = self;
        r.set_mul16();
        r
    }

    // Multiply this value by 32.
    #[inline(always)]
    pub fn set_mul32(&mut self) {
        // Extract the top bits
        let tt = self.0[1] >> 59;

        // Shift left
        self.0[1] = (self.0[1] << 5) | (self.0[0] >> 59);
        self.0[0] = self.0[0] << 5;

        // Wrap around the extracted bits
        self.set_reduce_small(tt);
    }

    #[inline(always)]
    pub fn mul32(self) -> Self {
        let mut r = self;
        r.set_mul32();
        r
    }

    // Multiply this value by a small integer.
    #[inline(always)]
    pub fn set_mul_small(&mut self, x: u32) {
        // Store as u64
        let b = x as u64;

        // Compute the product as an integer over three words.
        // Max value is (2^32 - 1)*(2^127 - 1), so the top word (d4) is
        // at most 2^31 - 1.
        let (d0, d1) = umull(self.0[0], b);
        let (lo, d2) = umull(self.0[1], b);
        let (d1, cc) = addcarry_u64(d1, lo, 0);
        let (d2, _) = addcarry_u64(d2, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.set_reduce_small(d2);
    }

    #[inline(always)]
    pub fn mul_small(self, x: u32) -> Self {
        let mut r = self;
        r.set_mul_small(x);
        r
    }

    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (rhs.0[0], rhs.0[1]);

        // Schoolbook Product
        let (e0, e1) = umull(a0, b0);
        let (e2, e3) = umull(a1, b1);

        let (lo, hi) = umull(a0, b1);
        let (lo2, hi2) = umull(a1, b0);
        let (lo, cc) = addcarry_u64(lo, lo2, 0);
        let (hi, tt) = addcarry_u64(hi, hi2, cc);
        let (e1, cc) = addcarry_u64(e1, lo, 0);
        let (e2, cc) = addcarry_u64(e2, hi, cc);
        let (e3, _) = addcarry_u64(e3, tt as u64, cc);

        // Reduction.
        // The maximum value if (2^127 - 1) * (2^127 - 1) = 2^254 - 2^128 + 1
        // Which we want to write as e = e0 || e1 || e2 || e3 = elo + 2^127 * ehi

        // elo is the lowest 127 bits of e
        let f0 = e0;
        let f1 = e1 & 0x7FFFFFFFFFFFFFFF;

        // ehi are top bits. As the maximum value is smaller than 2^254
        // the shift of e3 loses no information
        let g0 = (e1 >> 63) | (e2 << 1);
        let g1 = (e2 >> 63) | (e3 << 1);

        // We compute the output through addition of d = elo + ehi and then
        // reduce one last time
        let (d0, cc) = addcarry_u64(f0, g0, 0);
        let (d1, _) = addcarry_u64(f1, g1, cc);

        let m = (d1 >> 63) as u8;
        let d1 = d1 & 0x7FFFFFFFFFFFFFFF;
        let (d0, cc) = addcarry_u64(d0, 0, m);
        let (d1, _) = addcarry_u64(d1, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        // Compute the middle piece a0*a1
        let (a0, a1) = (self.0[0], self.0[1]);

        // 2. Double the intermediate value
        let (e1, e2) = umull(a0, a1);
        let e3 = e2 >> 63;
        let e2 = (e2 << 1) | (e1 >> 63);
        let e1 = e1 << 1;

        let (e0, hi) = umull(a0, a0);
        let (e1, cc) = addcarry_u64(e1, hi, 0);
        let (lo, hi) = umull(a1, a1);
        let (e2, cc) = addcarry_u64(e2, lo, cc);
        let (e3, _) = addcarry_u64(e3, hi, cc);

        // 3. Reduction (see set_mul() for details).
        let f0 = e0;
        let f1 = e1 & 0x7FFFFFFFFFFFFFFF;
        let g0 = (e1 >> 63) | (e2 << 1);
        let g1: u64 = (e2 >> 63) | (e3 << 1);

        let (d0, cc) = addcarry_u64(f0, g0, 0);
        let (d1, _) = addcarry_u64(f1, g1, cc);

        let m = (d1 >> 63) as u8;
        let d1 = d1 & 0x7FFFFFFFFFFFFFFF;
        let (d0, cc) = addcarry_u64(d0, 0, m);
        let (d1, _) = addcarry_u64(d1, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
    }

    // Square this value.
    #[inline(always)]
    pub fn square(self) -> Self {
        let mut r = self;
        r.set_square();
        r
    }

    // Square this value n times (in place).
    #[inline(always)]
    fn set_xsquare(&mut self, n: u32) {
        for _ in 0..n {
            self.set_square();
        }
    }

    // Square this value n times.
    #[inline(always)]
    pub fn xsquare(self, n: u32) -> Self {
        let mut r = self;
        r.set_xsquare(n);
        r
    }

    // The only issue we have is when self = 2^127 - 1
    // in which case we want to subtract p.
    // Here we check if self is the modulus and swap for
    // zero if this is true...
    // TODO: I think this is probably slow...
    #[inline]
    fn set_normalized(&mut self) {
        let t: u64 = (self.0[0] ^ Self::MODULUS[0]) | (self.0[1] ^ Self::MODULUS[1]);
        let r: u64 = t | t.wrapping_neg();
        let c: u32 = ((r >> 63) as u32).wrapping_sub(1);
        self.set_cond(&Self::ZERO, c);
    }

    // Compute a^((p-3)/4), used for inversion and Legendre
    // to compute a^(p-2) and a^((p-1)/2)
    #[inline(always)]
    fn set_exp_p_minus_three_div_four(&mut self) {
        let x = *self;
        let x2 = x * x.square();
        let x3 = x * x2.square();
        let x5 = x2 * x3.xsquare(2);
        let x10 = x5 * x5.xsquare(5);
        let x15 = x5 * x10.xsquare(5);
        let x30 = x15 * x15.xsquare(15);
        let x60 = x30 * x30.xsquare(30);
        let x120 = x60 * x60.xsquare(60);
        *self = x5 * x120.xsquare(5);
    }

    // TODO added
    pub fn set_invert(&mut self) {
        // from another file
        let mut x = *self;
        x.set_exp_p_minus_three_div_four();
        x.set_xsquare(2);
        *self *= x;
    }

    // TODO added
    pub fn invert(self) -> Self {
        let mut r = self;
        r.set_invert();
        r
    }

    // TODO added
    pub fn set_div(&mut self, y: &Self) {
        let y_inv = y.invert();
        self.set_mul(&y_inv)
    }

    // Perform a batch inversion of some elements. All elements of
    // the slice are replaced with their respective inverse (elements
    // of value zero are "inverted" into themselves).
    pub fn batch_invert(xx: &mut [Self]) {
        // We use Montgomery's trick:
        //   1/u = v*(1/(u*v))
        //   1/v = u*(1/(u*v))
        // Applied recursively on n elements, this computes an inversion
        // with a single inversion in the field, and 3*(n-1) multiplications.
        // We use batches of 200 elements; larger batches only yield
        // moderate improvements, while sticking to a fixed moderate batch
        // size allows stack-based allocation.
        let n = xx.len();
        let mut i = 0;
        while i < n {
            let blen = if (n - i) > 200 { 200 } else { n - i };
            let mut tt = [Self::ZERO; 200];
            tt[0] = xx[i];
            let zz0 = tt[0].iszero();
            tt[0].set_cond(&Self::ONE, zz0);
            for j in 1..blen {
                tt[j] = xx[i + j];
                tt[j].set_cond(&Self::ONE, tt[j].iszero());
                tt[j] *= tt[j - 1];
            }
            let mut k = Self::ONE / tt[blen - 1];
            for j in (1..blen).rev() {
                let mut x = xx[i + j];
                let zz = x.iszero();
                x.set_cond(&Self::ONE, zz);
                xx[i + j].set_cond(&(k * tt[j - 1]), !zz);
                k *= x;
            }
            xx[i].set_cond(&k, !zz0);
            i += blen;
        }
    }

    // Compute the Legendre symbol on this value. Return value is:
    //   0   if this value is zero
    //  +1   if this value is a non-zero quadratic residue
    //  -1   if this value is not a quadratic residue
    pub fn legendre(self) -> i32 {
        let mut l = self;
        l.set_exp_p_minus_three_div_four();
        l.set_square();
        l.set_mul(&self);

        let c1 = l.equals(&Self::ONE);
        let c2 = l.equals(&Self::MINUS_ONE);
        let cc1 = (c1 & 1) as i32;
        let cc2 = (c2 & 1) as i32;

        cc1 - cc2
    }

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). In the
    // latter case, this value is set to the square root of -self. In
    // all cases, the returned root is the one whose least significant
    // bit is 0 (when normalized in 0..q-1).
    fn set_sqrt_ext(&mut self) -> u32 {
        // Candidate root is self^((q+1)/4).
        // (q+1)/4 = 2^125
        let mut y = self.xsquare(125);

        // Normalize y and negate it if necessary to set the low bit to 0.
        // We must take care to make the bit check on the value in normal
        // representation, not Montgomery representation.
        let mut yn = y;
        yn.set_normalized();
        y.set_cond(&-y, ((yn.0[0] as u32) & 1).wrapping_neg());

        // Check that the candidate is indeed a square root.
        let r = y.square().equals(self);
        *self = y;
        r
    }

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). This
    // differs from set_sqrt_ext() in that this function sets the value
    // to zero if there is no square root.
    fn set_sqrt(&mut self) -> u32 {
        let r = self.set_sqrt_ext();
        self.set_cond(&Self::ZERO, !r);
        r
    }

    // Compute the square root of this value. Returned value are (y, r):
    //  - If this value is indeed a quadratic residue, then y is the
    //    square root whose least significant bit (when normalized in 0..q-1)
    //    is 0, and r is equal to 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is zero, and
    //    r is equal to 0.
    #[inline(always)]
    pub fn sqrt(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt();
        (x, r)
    }

    pub fn batch_sqrt<const N: usize>(inputs: &[Self; N]) -> [(Self, u32); N] {
        let mut y = *inputs;

        // Candidate root is self^((q+1)/4).
        // (q+1)/4 = 2^125
        for _ in 0..125 {
            for i in 0..N {
                y[i].set_square();
            }
        }

        let mut results = [(Self::ZERO, 0); N];
        for i in 0..N {
            // Normalize y and negate it if necessary to set the low bit to 0.
            let mut yn = y[i];
            yn.set_normalized();
            y[i].set_cond(&-y[i], ((yn.0[0] as u32) & 1).wrapping_neg());

            // Check that the candidate is indeed a square root.
            let r = y[i].square().equals(&inputs[i]);
            y[i].set_cond(&Self::ZERO, !r);
            results[i] = (y[i], r);
        }
        results
    }

    pub fn batch_invert_fixed<const N: usize>(inputs: &[Self; N]) -> [Self; N] {
        if N == 0 {
            return [Self::ZERO; N];
        }
        let mut tmp_inputs = *inputs;
        let mut is_zero_mask = [0u32; N];

        for i in 0..N {
            is_zero_mask[i] = tmp_inputs[i].iszero();
            tmp_inputs[i].set_cond(&Self::ONE, is_zero_mask[i]);
        }

        let mut scratch = [Self::ZERO; N];
        scratch[0] = tmp_inputs[0];
        for i in 1..N {
            scratch[i] = scratch[i - 1] * tmp_inputs[i];
        }

        let mut inv = scratch[N - 1].invert();

        let mut results = [Self::ZERO; N];
        for i in (1..N).rev() {
            let mut res = scratch[i - 1] * inv;
            res.set_cond(&Self::ZERO, is_zero_mask[i]);
            results[i] = res;
            inv = inv * tmp_inputs[i];
        }
        inv.set_cond(&Self::ZERO, is_zero_mask[0]);
        results[0] = inv;

        results
    }

    // Compute the square root of this value. Returned value are (y, r):
    //  - If this value is indeed a quadratic residue, then y is a
    //    square root of this value, and r is 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is set to
    //    a square root of -x, and r is 0x00000000.
    // In all cases, the returned root is normalized: the lest significant
    // bit of its integer representation (in the 0..q-1 range) is 0.
    #[inline(always)]
    pub fn sqrt_ext(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt_ext();
        (x, r)
    }

    /// Set this value to its fourth root. Returned value is 0xFFFFFFFF if
    /// the operation succeeded (value was indeed some element to the power of four), or
    /// 0x00000000 otherwise. On success, the chosen root is the one whose
    /// least significant bit (as an integer in [0..q-1]) is zero. On
    /// failure, this value is set to 0.
    fn set_fourth_root(&mut self) -> u32 {
        // Candidate root is self^((q+1)/8).
        // (q+1)/8 = 2^124
        let mut y = self.xsquare(124);

        // Normalize y and negate it if necessary to set the low bit to 0.
        // We must take care to make the bit check on the value in normal
        // representation, not Montgomery representation.
        let mut yn = y;
        yn.set_normalized();
        y.set_cond(&-y, ((yn.0[0] as u32) & 1).wrapping_neg());

        // Check that the candidate is indeed a square root.
        let r = y.xsquare(2).equals(self);
        *self = y;
        r
    }

    pub fn fourth_root(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_fourth_root();
        (x, r)
    }

    /// Set this value to its eighth root. Returned value is 0xFFFFFFFF if
    /// the operation succeeded (value was indeed some element to the power of eight), or
    /// 0x00000000 otherwise. On success, the chosen root is the one whose
    /// least significant bit (as an integer in [0..q-1]) is zero. On
    /// failure, this value is set to 0.
    fn set_eighth_root(&mut self) -> u32 {
        // Candidate root is self^((q+1)/16).
        // (q+1)/16 = 2^123
        let mut y = self.xsquare(123);

        // Normalize y and negate it if necessary to set the low bit to 0.
        // We must take care to make the bit check on the value in normal
        // representation, not Montgomery representation.
        let mut yn = y;
        yn.set_normalized();
        y.set_cond(&-y, ((yn.0[0] as u32) & 1).wrapping_neg());

        // Check that the candidate is indeed a square root.
        let r = y.xsquare(3).equals(self);
        *self = y;
        r
    }

    pub fn eighth_root(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_eighth_root();
        (x, r)
    }

    /// Set this structure to a random field element (indistinguishable
    /// from uniform generation).
    pub fn set_rand<T: CryptoRng + RngCore>(&mut self, rng: &mut T) {
        let mut tmp = [0u8; Self::ENCODED_LENGTH + 16];
        rng.fill_bytes(&mut tmp);
        self.set_decode_reduce(&tmp);
    }

    /// Return a new random field element (indistinguishable from
    /// uniform generation).
    pub fn rand<T: CryptoRng + RngCore>(rng: &mut T) -> Self {
        let mut x = Self::ZERO;
        x.set_rand(rng);
        x
    }

    /// Get the "hash" of the value (low 64 bits of the Montgomery
    /// representation).
    pub fn hashcode(self) -> u64 {
        self.0[0]
    }

    // Equality check between two field elements (constant-time);
    // returned value is 0xFFFFFFFF on equality, 0 otherwise.
    #[inline(always)]
    pub fn equals(self, rhs: &Self) -> u32 {
        (self - rhs).iszero()
    }

    // Compare this value with zero (constant-time); returned value
    // is 0xFFFFFFFF if this element is zero, 0 otherwise.
    #[inline(always)]
    pub fn iszero(self) -> u32 {
        // There are two possible representations for 0: 0 and q.
        let a0 = self.0[0];
        let a1 = self.0[1];
        let t0 = a0 | a1;
        let t1 = (a0 ^ Self::MODULUS[0]) | (a1 ^ Self::MODULUS[1]);

        // Top bit of r is 0 if and only if one of t0 or t1 is zero.
        let r = (t0 | t0.wrapping_neg()) & (t1 | t1.wrapping_neg());
        ((r >> 63) as u32).wrapping_sub(1)
    }

    // This internal function decodes 16 bytes (exactly) into a 128-bit
    // integer.
    #[inline(always)]
    fn set_decode16_raw(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 16);
        self.0[0] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[0..8]).unwrap());
        self.0[1] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[8..16]).unwrap());
    }

    // This internal function decodes 16 bytes (exactly) into a 128-bit
    // integer, and reduces it modulo q
    // WARNING: Montgomery representation is not applied.
    #[inline]
    fn set_decode16_reduce(&mut self, buf: &[u8]) {
        // Decode the 16 bytes as a raw integer.
        self.set_decode16_raw(buf);

        // Partial reduction.
        self.set_reduce_small(0);
    }

    // Encode this value over exactly 16 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..q-1 range.
    #[inline(always)]
    pub fn encode16(self) -> [u8; 16] {
        let mut r = self;
        r.set_normalized();
        let mut d = [0u8; 16];
        d[0..8].copy_from_slice(&r.0[0].to_le_bytes());
        d[8..16].copy_from_slice(&r.0[1].to_le_bytes());
        d
    }

    // Encode this value over exactly 16 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..q-1 range.
    #[inline(always)]
    pub fn encode(self) -> [u8; 16] {
        self.encode16()
    }

    // Decode the field element from the provided bytes. If the source
    // slice does not have length exactly 32 bytes, or if the encoding
    // is non-canonical (i.e. does not represent an integer in the 0
    // to q-1 range), then this element is set to zero, and 0 is returned.
    // Otherwise, this element is set to the decoded value, and 0xFFFFFFFF
    // is returned.
    #[inline]
    pub fn set_decode_ct(&mut self, buf: &[u8]) -> u32 {
        if buf.len() != 16 {
            *self = Self::ZERO;
            return 0;
        }

        self.set_decode16_raw(buf);

        // Try to subtract q from the value; if that does not yield a
        // borrow, then the encoding was not canonical.
        let (_, cc) = subborrow_u64(self.0[0], Self::MODULUS[0], 0);
        let (_, cc) = subborrow_u64(self.0[1], Self::MODULUS[1], cc);

        // Clear the value if not canonical.
        let cc = (cc as u64).wrapping_neg();
        self.0[0] &= cc;
        self.0[1] &= cc;

        cc as u32
    }

    // Decode a field element from 32 bytes. On success, this returns
    // (r, cc), where cc has value 0xFFFFFFFF. If the source encoding is not
    // canonical (i.e. the unsigned little-endian interpretation of the
    // 32 bytes yields an integer with is not lower than q), then this
    // returns (0, 0).
    #[inline(always)]
    pub fn decode(buf: &[u8]) -> (Self, u32) {
        let mut r = Self::ZERO;
        let cc = r.set_decode_ct(buf);
        (r, cc)
    }

    // Decode a field element from 32 bytes. On success, this returns
    // (r, cc), where cc has value 0xFFFFFFFF. If the source encoding is not
    // canonical (i.e. the unsigned little-endian interpretation of the
    // 32 bytes yields an integer with is not lower than q), then this
    // returns (0, 0).
    #[inline]
    pub fn decode16(buf: &[u8]) -> (Self, u32) {
        Self::decode(buf)
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer
    // is reduced modulo q. This process never fails.
    pub fn set_decode_reduce(&mut self, buf: &[u8]) {
        *self = Self::ZERO;
        let mut n = buf.len();
        if n == 0 {
            return;
        }
        if (n & 15) != 0 {
            // If input size is not a multiple of 16, then we decode a
            // partial block and the value is already less than 2^127.
            let k = n & !(15 as usize);
            let mut tmp = [0u8; 16];
            tmp[..(n - k)].copy_from_slice(&buf[k..]);
            n = k;
            self.set_decode16_raw(&tmp);
        } else {
            // If input size is a multiple of 16, then we decode a full
            // 16-byte block, and we must reduce it to get a 127-bit value.
            n -= 16;
            self.set_decode16_reduce(&buf[n..]);
        }

        // Process all remaining blocks, in descending address order.
        // TODO: Broken!
        while n > 0 {
            let k = n - 16;
            let mut v = Self::ZERO;
            v.set_decode16_reduce(&buf[k..k + 16]);
            self.set_add(&v);
            n = k;
        }
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer
    // is reduced modulo q. This process never fails.
    #[inline(always)]
    pub fn decode_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        r.set_decode_reduce(buf);
        r
    }

    // Partial reduction modulo q, as a function usable in constant
    // contexts. It is safe (constant-time) and thus also usable at
    // runtime, but less efficient than set_mul() since it does not
    // leverage intrinsics.
    const fn const_mred(a0: u64, a1: u64) -> [u64; 2] {
        // Custom add-with-carry.
        const fn adc(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128).wrapping_add(y as u128).wrapping_add(cc as u128);
            (z as u64, (z >> 64) as u64)
        }

        // Add the 127th bit to the lower word using that 2^127 = 1 mod p
        let m = a1 >> 63;
        let a1 = a1 & 0x7FFFFFFFFFFFFFFF;
        let (d0, cc) = adc(a0, 0, m);
        let (d1, cc) = adc(a1, 0, cc);
        assert!(cc == 0);

        // May need to do this twice
        let m = d1 >> 63;
        let d1 = d1 & 0x7FFFFFFFFFFFFFFF;
        let (d0, cc) = adc(d0, 0, m);
        let (d1, cc) = adc(d1, 0, cc);
        assert!(cc == 0);

        [d0, d1]
    }
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl Add<Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn add(self, other: Gf127) -> Gf127 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn add(self, other: &Gf127) -> Gf127 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn add(self, other: Gf127) -> Gf127 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn add(self, other: &Gf127) -> Gf127 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl AddAssign<Gf127> for Gf127 {
    #[inline(always)]
    fn add_assign(&mut self, other: Gf127) {
        self.set_add(&other);
    }
}

impl AddAssign<&Gf127> for Gf127 {
    #[inline(always)]
    fn add_assign(&mut self, other: &Gf127) {
        self.set_add(other);
    }
}

impl Div<Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn div(self, other: Gf127) -> Gf127 {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl Div<&Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn div(self, other: &Gf127) -> Gf127 {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl Div<Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn div(self, other: Gf127) -> Gf127 {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl Div<&Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn div(self, other: &Gf127) -> Gf127 {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl DivAssign<Gf127> for Gf127 {
    #[inline(always)]
    fn div_assign(&mut self, other: Gf127) {
        self.set_div(&other);
    }
}

impl DivAssign<&Gf127> for Gf127 {
    #[inline(always)]
    fn div_assign(&mut self, other: &Gf127) {
        self.set_div(other);
    }
}

impl Mul<Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn mul(self, other: Gf127) -> Gf127 {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn mul(self, other: &Gf127) -> Gf127 {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn mul(self, other: Gf127) -> Gf127 {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn mul(self, other: &Gf127) -> Gf127 {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<Gf127> for Gf127 {
    #[inline(always)]
    fn mul_assign(&mut self, other: Gf127) {
        self.set_mul(&other);
    }
}

impl MulAssign<&Gf127> for Gf127 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &Gf127) {
        self.set_mul(other);
    }
}

impl Neg for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn neg(self) -> Gf127 {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl Neg for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn neg(self) -> Gf127 {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl Sub<Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn sub(self, other: Gf127) -> Gf127 {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&Gf127> for Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn sub(self, other: &Gf127) -> Gf127 {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl Sub<Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn sub(self, other: Gf127) -> Gf127 {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&Gf127> for &Gf127 {
    type Output = Gf127;

    #[inline(always)]
    fn sub(self, other: &Gf127) -> Gf127 {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl SubAssign<Gf127> for Gf127 {
    #[inline(always)]
    fn sub_assign(&mut self, other: Gf127) {
        self.set_sub(&other);
    }
}

impl SubAssign<&Gf127> for Gf127 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &Gf127) {
        self.set_sub(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::Gf127;
    use core::convert::TryFrom;
    use num_bigint::{BigInt, Sign};
    use sha2::{Digest, Sha256};

    // fn print(name: &str, v: Gf127) {
    //     println!("{} = 0x{:016X}{:016X}", name, v.0[1], v.0[0]);
    // }

    // va, vb and vx must be 16 bytes each in length
    fn check_gf_ops(va: &[u8], vb: &[u8]) {
        let zp = BigInt::from_slice(
            Sign::Plus,
            &[0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0x7FFFFFFFu32],
        );
        let zpz = &zp << 64;
        let zp8 = &zp << 8;

        let mut a = Gf127::ZERO;
        a.set_decode_reduce(va);
        let mut b = Gf127::ZERO;
        b.set_decode_reduce(vb);
        let za = BigInt::from_bytes_le(Sign::Plus, va);
        let zb = BigInt::from_bytes_le(Sign::Plus, vb);
        let vc = a.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;
        assert!(zc == zd);

        let a0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&va[0..8]).unwrap());
        let a1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&va[8..16]).unwrap());
        let c = Gf127::w64le(a0, a1);
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;

        assert!(zc == zd);
        let c = Gf127::w64be(a1, a0);
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;
        assert!(zc == zd);

        let c = Gf127::from_w64le(a0, a1);
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;
        assert!(zc == zd);

        let c = Gf127::from_w64be(a1, a0);
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;
        assert!(zc == zd);

        let c = a + b;
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za + &zb) % &zp;
        assert!(zc == zd);

        let c = a - b;
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = ((&zp8 + &za) - &zb) % &zp;
        assert!(zc == zd);

        let c = -a;
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&zp8 - &za) % &zp;
        assert!(zc == zd);

        let c = a * b;
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &zb) % &zp;
        assert!(zc == zd);

        let c = a.half();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd: BigInt = ((&zp8 + (&zc << 1)) - &za) % &zp;
        assert!(zd.sign() == Sign::NoSign);

        let c = a.mul2();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 1) % &zp;
        assert!(zc == zd);

        let c = a.mul4();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 2) % &zp;
        assert!(zc == zd);

        let c = a.mul8();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 3) % &zp;
        assert!(zc == zd);

        let c = a.mul16();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 4) % &zp;
        assert!(zc == zd);

        let c = a.mul32();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 5) % &zp;
        assert!(zc == zd);

        let x = u32::from_le_bytes(*<&[u8; 4]>::try_from(&vb[0..4]).unwrap());
        let c = a.mul_small(x);
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = ((&za % &zp) * x + &zpz) % &zp;
        assert!(zc == zd);

        let c = a.square();
        let vc = c.encode16();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &za) % &zp;
        assert!(zc == zd);

        let (e, cc) = Gf127::decode16(va);
        if cc != 0 {
            assert!(cc == 0xFFFFFFFF);
            assert!(e.encode16() == va);
        } else {
            assert!(e.encode16() == [0u8; 16]);
        }

        let c = a / b;
        let d = c * b;
        if b.iszero() != 0 {
            assert!(c.iszero() != 0);
        } else {
            assert!(a.equals(&d) != 0);
        }
    }

    #[test]
    fn Gf127_ops() {
        let mut va = [0u8; 16];
        let mut vb = [0u8; 16];
        check_gf_ops(&va, &vb);
        assert!(Gf127::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        assert!(Gf127::decode_reduce(&va).equals(&Gf127::decode_reduce(&vb)) == 0xFFFFFFFF);
        assert!(Gf127::decode_reduce(&va).legendre() == 0);
        for i in 0..16 {
            va[i] = 0xFFu8;
            vb[i] = 0xFFu8;
        }
        check_gf_ops(&va, &vb);
        assert!(Gf127::decode_reduce(&va).iszero() == 0);
        assert!(Gf127::decode_reduce(&va).equals(&Gf127::decode_reduce(&vb)) == 0xFFFFFFFF);
        for i in 0..2 {
            va[8 * i..8 * i + 8].copy_from_slice(&Gf127::MODULUS[i].to_le_bytes());
        }
        assert!(Gf127::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        let mut sh = Sha256::new();
        for i in 0..1000 {
            sh.update(((2 * i + 0) as u64).to_le_bytes());
            let va = &sh.finalize_reset()[0..16];
            sh.update(((2 * i + 1) as u64).to_le_bytes());
            let vb = &sh.finalize_reset()[0..16];
            check_gf_ops(&va, &vb);
            assert!(Gf127::decode_reduce(&va).iszero() == 0);
            assert!(Gf127::decode_reduce(&va).equals(&Gf127::decode_reduce(&vb)) == 0);
            let s = Gf127::decode_reduce(&va).square();
            let s2 = -s;
            assert!(s.legendre() == 1);
            assert!(s2.legendre() == -1);
            let (t, r) = s.sqrt();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().equals(&s) == 0xFFFFFFFF);
            assert!((t.encode16()[0] & 1) == 0);
            let (t, r) = s.sqrt_ext();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().equals(&s) == 0xFFFFFFFF);
            assert!((t.encode16()[0] & 1) == 0);
            let (t2, r) = s2.sqrt();
            assert!(r == 0);
            assert!(t2.iszero() == 0xFFFFFFFF);
            let (t2, r) = s2.sqrt_ext();
            assert!(r == 0);
            assert!(t2.square().equals(&-s2) == 0xFFFFFFFF);

            // test fourth root
            let s = Gf127::decode_reduce(&va).square().square();
            let (t, r) = s.fourth_root();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().square().equals(&s) == 0xFFFFFFFF);

            // test eighth root
            let s = Gf127::decode_reduce(&va).square().square().square();
            let (t, r) = s.eighth_root();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().square().square().equals(&s) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn Gf127_batch_invert() {
        let mut xx = [Gf127::ZERO; 300];
        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update((i as u64).to_le_bytes());
            let v = &sh.finalize_reset()[0..16];
            xx[i] = Gf127::decode_reduce(&v);
        }
        xx[120] = Gf127::ZERO;
        let mut yy = xx;
        Gf127::batch_invert(&mut yy[..]);
        for i in 0..300 {
            if xx[i].iszero() != 0 {
                assert!(yy[i].iszero() == 0xFFFFFFFF);
            } else {
                assert!((xx[i] * yy[i]).equals(&Gf127::ONE) == 0xFFFFFFFF);
            }
        }
    }

    #[test]
    fn Gf127_batch_invert_fixed() {
        let mut prng = Sha256::new();
        for _ in 0..10 {
            let mut inputs = [Gf127::ZERO; 6];
            for i in 0..6 {
                prng.update((i as u64).to_le_bytes());
                inputs[i] = Gf127::decode_reduce(&prng.finalize_reset());
            }
            // Add a zero
            inputs[2] = Gf127::ZERO;

            let results = Gf127::batch_invert_fixed::<6>(&inputs);
            for i in 0..6 {
                let expected = inputs[i].invert();
                assert!(results[i].equals(&expected) == 0xFFFFFFFF);
            }
        }
    }

    #[test]
    fn Gf127_batch_sqrt() {
        let mut prng = Sha256::new();
        for _ in 0..10 {
            let mut inputs = [Gf127::ZERO; 6];
            for i in 0..6 {
                prng.update((i as u64).to_le_bytes());
                // Square to ensure quadratic residue
                inputs[i] = Gf127::decode_reduce(&prng.finalize_reset()).square();
            }
            // Add a zero
            inputs[4] = Gf127::ZERO;

            let results = Gf127::batch_sqrt::<6>(&inputs);
            for i in 0..6 {
                let (expected_res, expected_r) = inputs[i].sqrt();
                let (actual_res, actual_r) = results[i];
                assert_eq!(expected_r, actual_r);
                assert!(actual_res.equals(&expected_res) == 0xFFFFFFFF);
            }
        }
    }
}
