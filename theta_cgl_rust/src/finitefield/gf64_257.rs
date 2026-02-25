use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use rand_core::{CryptoRng, RngCore};

// ========================================================================
// GF(p)

/// An element of GF(p).
#[derive(Clone, Copy, Debug)]
pub struct GFp(u64);

impl GFp {
    // IMPLEMENTATION NOTES:
    // ---------------------
    //
    // An element x is lazily reduced within the range [0, 2^64 - 1]
    // to save a cycle or two during reduction
    //
    // Everything is constant-time. There are specialized "Boolean"
    // functions such as iszero() and equals() that output a u32 which
    // happens, in practice, to have value 0 (for "false") or 2^32-1
    // (for "true").

    // Some constants needed for the Fp2 macro
    pub const ENCODED_LENGTH: usize = 8;
    pub const N: usize = 1;

    /// GF(p) modulus: p = 2^64 - 2^8 - 1
    pub const MOD: u64 = 0xfffffffffffffeff;
    pub const MODULUS: [u64; GFp::N] = [0xFFFFFFFFFFFFFEFF];

    /// Element 0 in GF(p).
    pub const ZERO: GFp = GFp::from_u64_reduce(0);

    /// Element 1 in GF(p).
    pub const ONE: GFp = GFp::from_u64_reduce(1);

    /// Element -1 in GF(p).
    pub const MINUS_ONE: GFp = GFp::from_u64_reduce(GFp::MOD - 1);

    pub const TWO: Self = GFp::from_u64_reduce(2);
    pub const THREE: Self = GFp::from_u64_reduce(3);
    pub const FOUR: Self = GFp::from_u64_reduce(4);

    const P_PLUS_ONE_HALF: u64 = 0x7fffffffffffff80;

    // Full reduction modulo p = 2^64 - 2^8 - 1
    // produces a canonical result in [0, p-1]
    #[inline(always)]
    const fn fp_normalized(x: u64) -> u64 {
        // Subtract the modulus and add it back if there was an
        // overflow
        let (r, cc) = x.overflowing_sub(GFp::MOD);
        let m = (cc as u64).wrapping_neg();
        r.wrapping_add(Self::MOD & m)
    }

    /// Build a GF(p) element from a 64-bit integer. Returned values
    /// are (r, c). If the source value v is lower than the modulus,
    /// then r contains the value v as an element of GF(p), and c is
    /// equal to 0xFFFFFFFFFFFFFFFF; otherwise, r contains zero (in
    /// GF(p)) and c is 0.
    #[inline(always)]
    pub fn from_u64(v: u64) -> (Self, u64) {
        // Computation of c is a constant-time lower-than operation:
        // If v < 2^63 then v < p and its high bit is 0.
        // If v >= 2^63 then its high bit is 1, and v < p if and only if
        // the high bit of z = v-p is 1 (since p >= 2^63 itself).
        let z = v.wrapping_sub(GFp::MOD);
        let c = ((v & !z) >> 63).wrapping_sub(1);
        (GFp::from_u64_reduce(v & c), c)
    }

    /// Build a GF(p) element from a 64-bit integer. The provided
    /// integer is implicitly reduced modulo p.
    #[inline(always)]
    pub const fn from_u64_reduce(v: u64) -> Self {
        GFp(GFp::fp_normalized(v))
    }

    /// Get the element as an integer, normalized in the 0..p-1
    /// range.
    #[inline(always)]
    pub const fn to_u64(self) -> u64 {
        GFp::fp_normalized(self.0)
    }

    /// Addition in GF(p)
    #[inline(always)]
    const fn add(self, rhs: Self) -> Self {
        let (x1, c1) = self.0.overflowing_add(rhs.0);
        let t = (c1 as u64) + ((c1 as u64) << 8);
        GFp(x1.wrapping_add(t as u64))
    }

    /// Subtraction in GF(p)
    #[inline(always)]
    const fn sub(self, rhs: Self) -> Self {
        let (x1, cc) = self.0.overflowing_sub(rhs.0);
        let m = (cc as u64).wrapping_neg();
        GFp(x1.wrapping_add(m & Self::MOD))
    }

    /// Negation in GF(p)
    #[inline(always)]
    const fn neg(self) -> Self {
        let (x1, cc) = Self::MOD.overflowing_sub(self.0);
        let m = (cc as u64).wrapping_neg();
        GFp(x1.wrapping_add(m & Self::MOD))
    }

    /// Halving in GF(p) (division by 2).
    #[inline(always)]
    pub const fn half(self) -> Self {
        // Right shift the value by one bit
        // If x is even, then this returned x/2.
        // If x is odd, then this returns (x-1)/2 + (p+1)/2 = (x+p)/2.
        GFp((self.0 >> 1) + ((self.0 & 1).wrapping_neg() & Self::P_PLUS_ONE_HALF))
    }

    /// Doubling in GF(p) (multiplication by 2).
    #[inline(always)]
    pub const fn double(self) -> Self {
        self.add(self)
    }

    /// Multiplication in GF(p)
    #[inline(always)]
    const fn mul(self, rhs: Self) -> Self {
        // Internally all values are in the range [0, 2^64]
        // Double wide multiplication of two u64 fits in a u128
        let x = (self.0 as u128) * (rhs.0 as u128);

        // Now perform partial reduction
        //
        // x = x_lo + 2^64 * x_hi
        //   = x_lo + x_hi + 2^8*x_hi
        //
        // The resulting v will have 64 + 9 bits max
        let x_lo = x as u64;
        let x_hi = x >> 64;
        let v = (x_lo as u128) + x_hi + (x_hi << 8);

        // Now we need to fold in these pieces where cc < 2^9
        let v_lo = v as u64;
        let v_hi = (v >> 64) as u64;
        let (r, cc) = v_lo.overflowing_add(v_hi + (v_hi << 8));

        // We need to fold in one last time
        let d = cc as u64;
        GFp(r + d + (d << 8))
    }

    /// Squaring in GF(p)
    #[inline(always)]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiple squarings in GF(p): return x^(2^n)
    #[inline(always)]
    pub fn xsquare(self, n: u32) -> Self {
        let mut x = self;
        for _ in 0..n {
            x = x.square();
        }
        x
    }

    /// Inversion in GF(p); if the input is zero, then zero is returned.
    #[inline(always)]
    pub fn invert(self) -> Self {
        // This uses Fermat's little theorem: 1/x = x^(p-2) mod p.
        let x = self.exp_p_minus_three_div_four();
        self * x.xsquare(2)
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert()
    }

    /// Test of equality with zero; return value is 0xFFFFFFFF
    /// if this value is equal to zero, or 0 otherwise.
    #[inline(always)]
    pub const fn iszero(self) -> u32 {
        // As we use a non-canonical representation, then the zero
        // value can be either 0 or p.
        let t0 = self.0;
        let t1 = self.0 ^ Self::MOD;

        // Top bit of r is 0 if and only if one of t0 or t1 is zero.
        let r = (t0 | t0.wrapping_neg()) & (t1 | t1.wrapping_neg());
        ((r >> 63) as u32).wrapping_sub(1)
    }

    /// Test of equality between two GF(p) elements; return value is
    /// 0xFFFFFFFF if the two values are equal, or 0 otherwise.
    #[inline(always)]
    pub fn equals(self, rhs: &Self) -> u32 {
        (self - rhs).iszero()
    }

    // Compute a^((p-3)/4), used for inversion and Legendre
    // to compute a^(p-2) and a^((p-1)/2)
    fn exp_p_minus_three_div_four(self) -> Self {
        let x = self;
        let x2 = x * x.square(); // x^(2^2 - 1)
        let x3 = x * x2.square(); // x^(2^3 - 1)
        let x6 = x3 * x3.xsquare(3); // x^(2^6 - 1)
        let x12 = x6 * x6.xsquare(6); // x^(2^12 - 1)
        let x24 = x12 * x12.xsquare(12); // x^(2^24 - 1)
        let x48 = x24 * x24.xsquare(24); // x^(2^48 - 1)
        let x54 = x6 * x48.xsquare(6); // x^(2^54 - 1)
        let x55 = x * x54.square(); // x^(2^55 - 1)
        x6 * x55.xsquare(7) // x^(2^62 - 2**7) + (2^6 - 1) =
                            // x^(2^62 - 2**6 - 1) = (p-3)/4
    }

    pub fn legendre(self) -> i32 {
        let mut l = self.exp_p_minus_three_div_four();
        l = self * l.square();

        let c1 = l.equals(&GFp::ONE);
        let c2 = l.equals(&GFp::MINUS_ONE);
        let cc1 = (c1 & 1) as i32;
        let cc2 = (c2 & 1) as i32;

        cc1 - cc2
    }

    pub fn sqrt(self) -> (Self, u32) {
        // We use that p = 3 mod 4 to compute the square root by
        // x^((p+1)/4) mod p. We have (p+1)/4 = 2^62 - 2^6
        // In the instructions below, we call 'xj' the value x^(2^j-1).
        let x = self;
        let x2 = x * x.square();
        let x4 = x2 * x2.xsquare(2);
        let x6 = x2 * x4.xsquare(2);
        let x7 = x * x6.square();
        let x14 = x7 * x7.xsquare(7);
        let x28 = x14 * x14.xsquare(14);
        let x56 = x28 * x28.xsquare(28);
        let mut y = x56.xsquare(6);

        let ctl = ((y.0 as u32) & 1).wrapping_neg();
        y.set_condneg(ctl);

        let r = y.square().equals(&self);
        let r_64 = (r as u64) | ((r as u64) << 32);
        y.0 &= r_64;

        (y, r)
    }

    pub fn batch_sqrt<const N: usize>(inputs: &[Self; N]) -> [(Self, u32); N] {
        let x = *inputs;

        // Helper to square all N elements
        let batch_square = |v: &mut [Self; N]| {
            for i in 0..N {
                v[i] = v[i].square();
            }
        };

        // Helper to multiply two arrays of N elements
        let batch_mul = |dest: &mut [Self; N], src: &[Self; N]| {
            for i in 0..N {
                dest[i] = dest[i] * src[i];
            }
        };

        // Helper for xsquare (repeated squaring)
        let batch_xsquare = |v: &mut [Self; N], n: u32| {
            for _ in 0..n {
                batch_square(v);
            }
        };

        // x2 = x * x.square()
        let mut t = x;
        batch_square(&mut t);
        let mut x2 = x;
        batch_mul(&mut x2, &t);

        // x4 = x2 * x2.xsquare(2)
        t = x2;
        batch_xsquare(&mut t, 2);
        let mut x4 = x2;
        batch_mul(&mut x4, &t);

        // x6 = x2 * x4.xsquare(2)
        t = x4;
        batch_xsquare(&mut t, 2);
        let mut x6 = x2;
        batch_mul(&mut x6, &t);

        // x7 = x * x6.square()
        t = x6;
        batch_square(&mut t);
        let mut x7 = x;
        batch_mul(&mut x7, &t);

        // x14 = x7 * x7.xsquare(7)
        t = x7;
        batch_xsquare(&mut t, 7);
        let mut x14 = x7;
        batch_mul(&mut x14, &t);

        // x28 = x14 * x14.xsquare(14)
        t = x14;
        batch_xsquare(&mut t, 14);
        let mut x28 = x14;
        batch_mul(&mut x28, &t);

        // x56 = x28 * x28.xsquare(28)
        t = x28;
        batch_xsquare(&mut t, 28);
        let mut x56 = x28;
        batch_mul(&mut x56, &t);

        // y = x56.xsquare(6)
        let mut y = x56;
        batch_xsquare(&mut y, 6);

        let mut results = [(Self::ZERO, 0); N];

        for i in 0..N {
            let ctl = ((y[i].0 as u32) & 1).wrapping_neg();
            y[i].set_condneg(ctl);

            let r = y[i].square().equals(&inputs[i]);
            let r_64 = (r as u64) | ((r as u64) << 32);
            y[i].0 &= r_64;
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

    /// Set this value to its fourth root. Returned value is 0xFFFFFFFF if
    /// the operation succeeded (value was indeed some element to the power of four), or
    /// 0x00000000 otherwise. On success, the chosen root is the one whose
    /// least significant bit (as an integer in [0..p-1]) is zero. On
    /// failure, this value is set to 0.
    pub fn fourth_root(self) -> (Self, u32) {
        let x = self;
        let x2 = x * x.square();
        let x4 = x2 * x2.xsquare(2);
        let x6 = x2 * x4.xsquare(2);
        let x7 = x * x6.square();
        let x14 = x7 * x7.xsquare(7);
        let x28 = x14 * x14.xsquare(14);
        let x56 = x28 * x28.xsquare(28);
        let mut y = x56.xsquare(5);

        // Check that the obtained value is indeed a fourth root of the
        // source value (which is still in ww[0]); if not, clear this
        // value.
        let r = y.square().square().equals(&x);
        let rw = (r as u64) | ((r as u64) << 32);
        y.0 &= rw;

        // Conditionally negate this value, so that the chosen root
        // follows the expected convention.
        let ctl = ((y.0 as u32) & 1).wrapping_neg();

        y.set_condneg(ctl);

        (y, r)
    }

    /// Set this value to its eighth root. Returned value is 0xFFFFFFFF if
    /// the operation succeeded (value was indeed some element to the power of eight), or
    /// 0x00000000 otherwise. On success, the chosen root is the one whose
    /// least significant bit (as an integer in [0..p-1]) is zero. On
    /// failure, this value is set to 0.
    pub fn eighth_root(self) -> (Self, u32) {
        let x = self;
        let x2 = x * x.square();
        let x4 = x2 * x2.xsquare(2);
        let x6 = x2 * x4.xsquare(2);
        let x7 = x * x6.square();
        let x14 = x7 * x7.xsquare(7);
        let x28 = x14 * x14.xsquare(14);
        let x56 = x28 * x28.xsquare(28);
        let mut y = x56.xsquare(4);

        // Check that the obtained value is indeed a eighth root of the
        // source value (which is still in ww[0]); if not, clear this
        // value.
        let r = y.xsquare(3).equals(&x);
        let rw = (r as u64) | ((r as u64) << 32);
        y.0 &= rw;

        // Conditionally negate this value, so that the chosen root
        // follows the expected convention.
        let ctl = ((y.0 as u32) & 1).wrapping_neg();

        y.set_condneg(ctl);

        (y, r)
    }

    /// Select a value: this function returns x0 if c == 0, or x1 if
    /// c == 0xFFFFFFFF.
    #[inline(always)]
    pub fn select(x0: &Self, x1: &Self, c: u32) -> Self {
        let c_64 = (c as u64) | ((c as u64) << 32);
        GFp(x0.0 ^ (c_64 & (x0.0 ^ x1.0)))
    }

    /// Negate this value.
    #[inline(always)]
    pub fn set_neg(&mut self) {
        self.0 = self.neg().0;
    }

    /// Multiplication in GF(p) by a small integer (less than 2^31).
    #[inline(always)]
    pub fn mul_small(self, rhs: u32) -> Self {
        // As the rhs size is bounded, we can do a slightly cheaper
        // reduction modulo p
        let x = (self.0 as u128) * (rhs as u128);
        let xl = x as u64;
        let xh = (x >> 64) as u64;
        let (r, cc) = xl.overflowing_add(xh + (xh << 8));
        let d = cc as u64;
        GFp(r + d + (d << 8))
    }

    #[inline(always)]
    pub fn set_mul_small(&mut self, rhs: u32) {
        let r = self.mul_small(rhs);
        self.0 = r.0;
    }

    /// Double this value.
    #[inline(always)]
    pub fn set_mul2(&mut self) {
        let r = self.double();
        self.0 = r.0;
    }

    /// Compute the sum of this value with itself.
    #[inline(always)]
    pub fn mul2(self) -> Self {
        let mut r = self;
        r.set_mul2();
        r
    }

    /// Compute the quadruple of this value.
    #[inline(always)]
    pub fn mul4(self) -> Self {
        let mut r = self;
        r.set_mul4();
        r
    }

    /// Halve this value.
    #[inline]
    pub fn set_half(&mut self) {
        self.0 = self.half().0;
    }

    /// Quadruple this value.
    #[inline]
    pub fn set_mul4(&mut self) {
        self.set_mul2();
        self.set_mul2();
    }

    /// Multiply this value by 8
    #[inline]
    pub fn set_mul8(&mut self) {
        self.set_mul2();
        self.set_mul2();
        self.set_mul2();
    }

    /// Compute the quadruple of this value.
    #[inline(always)]
    pub fn mul8(self) -> Self {
        let mut r = self;
        r.set_mul8();
        r
    }

    /// Set this value to either a or b, depending on whether the control
    /// word ctl is 0x00000000 or 0xFFFFFFFF, respectively.
    /// The value of ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline]
    pub fn set_select(&mut self, a: &Self, b: &Self, ctl: u32) {
        self.0 = GFp::select(a, b, ctl).0;
    }

    /// Set this value to rhs if ctl is 0xFFFFFFFF; leave it unchanged if
    /// ctl is 0x00000000.
    /// The value of ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline]
    pub fn set_cond(&mut self, rhs: &Self, ctl: u32) {
        let wa = self.0;
        let wb = rhs.0;
        let ctl_64 = (ctl as u64) | ((ctl as u64) << 32);
        self.0 = wa ^ (ctl_64 & (wa ^ wb));
    }

    /// Exchange the values of a and b is ctl is 0xFFFFFFFF; leave both
    /// values unchanged if ctl is 0x00000000.
    /// The value of ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline]
    pub fn condswap(a: &mut Self, b: &mut Self, c: u32) {
        let wa = a.0;
        let wb = b.0;
        let c_64 = (c as u64) | ((c as u64) << 32);
        let wc = c_64 & (wa ^ wb);
        a.0 = wa ^ wc;
        b.0 = wb ^ wc;
    }

    #[inline]
    pub fn set_invert(&mut self) {
        self.0 = self.invert().0;
    }

    /// Negate this value if ctl is 0xFFFFFFFF; leave it unchanged if
    /// ctl is 0x00000000.
    /// The value of ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline]
    pub fn set_condneg(&mut self, ctl: u32) {
        let v = self.neg();
        self.set_cond(&v, ctl);
    }

    pub fn encode(self) -> [u8; Self::ENCODED_LENGTH] {
        let r = self.to_u64();
        let mut d = [0u8; Self::ENCODED_LENGTH];
        d[0..].copy_from_slice(&(r.to_le_bytes()[..Self::ENCODED_LENGTH]));
        d
    }

    pub fn decode(buf: &[u8]) -> (Self, u64) {
        if buf.len() != Self::ENCODED_LENGTH {
            return (GFp::ZERO, 0);
        }
        GFp::from_u64(u64::from_le_bytes(
            *<&[u8; 8]>::try_from(&buf[0..8]).unwrap(),
        ))
    }

    /// Get the "hash" of the value
    pub fn hashcode(self) -> u64 {
        self.0
    }

    /// Decode the provided bytes. The source slice
    /// can have arbitrary length; the bytes are interpreted with the
    /// unsigned little-endian convention (no sign bit), and the resulting
    /// integer is reduced modulo the field modulus p. By definition, this
    /// function does not enforce canonicality of the source value.
    #[inline]
    pub fn decode_reduce(buf: &[u8]) -> Self {
        let mut n = buf.len();
        let mut x = Self::ZERO;
        if n == 0 {
            return x;
        }

        let mut tmp = [0u8; Self::ENCODED_LENGTH];
        const CLEN: usize = 8;
        let mut nn = n % CLEN;
        if nn == 0 {
            nn = CLEN;
        }
        n -= nn;
        tmp[..nn].copy_from_slice(&buf[n..]);
        (x, _) = GFp::decode(&tmp);

        while n > 0 {
            n -= CLEN;
            tmp[..CLEN].copy_from_slice(&buf[n..(n + CLEN)]);
            let (d, _) = GFp::decode(&tmp);
            x = x + d;
        }

        x
    }

    pub fn set_decode_reduce(&mut self, buf: &[u8]) {
        let r = Self::decode_reduce(buf);
        self.0 = r.0;
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
}

// We implement all the needed traits to allow use of the arithmetic
// operators on GF(p) values.

impl Add<GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn add(self, other: GFp) -> GFp {
        GFp::add(self, other)
    }
}

impl Add<&GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn add(self, other: &GFp) -> GFp {
        GFp::add(self, *other)
    }
}

impl Add<GFp> for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn add(self, other: GFp) -> GFp {
        GFp::add(*self, other)
    }
}

impl Add<&GFp> for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn add(self, other: &GFp) -> GFp {
        GFp::add(*self, *other)
    }
}

impl AddAssign<GFp> for GFp {
    #[inline(always)]
    fn add_assign(&mut self, other: GFp) {
        *self = GFp::add(*self, other);
    }
}

impl AddAssign<&GFp> for GFp {
    #[inline(always)]
    fn add_assign(&mut self, other: &GFp) {
        *self = GFp::add(*self, *other);
    }
}

impl Sub<GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn sub(self, other: GFp) -> GFp {
        GFp::sub(self, other)
    }
}

impl Sub<&GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn sub(self, other: &GFp) -> GFp {
        GFp::sub(self, *other)
    }
}

impl Sub<GFp> for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn sub(self, other: GFp) -> GFp {
        GFp::sub(*self, other)
    }
}

impl Sub<&GFp> for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn sub(self, other: &GFp) -> GFp {
        GFp::sub(*self, *other)
    }
}

impl SubAssign<GFp> for GFp {
    #[inline(always)]
    fn sub_assign(&mut self, other: GFp) {
        *self = GFp::sub(*self, other);
    }
}

impl SubAssign<&GFp> for GFp {
    #[inline(always)]
    fn sub_assign(&mut self, other: &GFp) {
        *self = GFp::sub(*self, *other);
    }
}

impl Neg for GFp {
    type Output = GFp;

    #[inline(always)]
    fn neg(self) -> GFp {
        GFp::neg(self)
    }
}

impl Neg for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn neg(self) -> GFp {
        GFp::neg(*self)
    }
}

impl Mul<GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn mul(self, other: GFp) -> GFp {
        GFp::mul(self, other)
    }
}

impl Mul<&GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn mul(self, other: &GFp) -> GFp {
        GFp::mul(self, *other)
    }
}

impl Mul<GFp> for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn mul(self, other: GFp) -> GFp {
        GFp::mul(*self, other)
    }
}

impl Mul<&GFp> for &GFp {
    type Output = GFp;

    #[inline(always)]
    fn mul(self, other: &GFp) -> GFp {
        GFp::mul(*self, *other)
    }
}

impl MulAssign<GFp> for GFp {
    #[inline(always)]
    fn mul_assign(&mut self, other: GFp) {
        *self = GFp::mul(*self, other);
    }
}

impl MulAssign<&GFp> for GFp {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GFp) {
        *self = GFp::mul(*self, *other);
    }
}

impl Div<GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn div(self, other: GFp) -> GFp {
        GFp::div(self, other)
    }
}

impl Div<&GFp> for GFp {
    type Output = GFp;

    #[inline(always)]
    fn div(self, other: &GFp) -> GFp {
        GFp::div(self, *other)
    }
}

impl DivAssign<GFp> for GFp {
    #[inline(always)]
    fn div_assign(&mut self, other: GFp) {
        *self = GFp::div(*self, other);
    }
}

impl DivAssign<&GFp> for GFp {
    #[inline(always)]
    fn div_assign(&mut self, other: &GFp) {
        *self = GFp::div(*self, *other);
    }
}

// ========================================================================
// Unit tests.

#[cfg(test)]
mod tests {
    use super::GFp;

    // A custom PRNG; not cryptographically secure, but good enough
    // for tests.
    #[cfg(test)]
    struct PRNG(u128);

    #[cfg(test)]
    impl PRNG {
        // A: a randomly selected prime integer.
        // B: a randomly selected odd integer.
        const A: u128 = 87981536952642681582438141175044346919;
        const B: u128 = 331203846847999889118488772711684568729;

        // Get the next pseudo-random 64-bit integer.
        fn next_u64(&mut self) -> u64 {
            self.0 = PRNG::A.wrapping_mul(self.0).wrapping_add(PRNG::B);
            (self.0 >> 64) as u64
        }
    }

    fn check_gfp_eq(a: GFp, r: u128) {
        assert!(a.to_u64() == (r % (GFp::MOD as u128)) as u64);
    }

    fn test_gfp_ops(a: u64, b: u64) {
        let x = GFp::from_u64_reduce(a);
        let y = GFp::from_u64_reduce(b);
        let wa = a as u128;
        let wb = b as u128;
        let small_b = (b & 0xFFFF) as u128;

        check_gfp_eq(x + y, wa + wb);
        check_gfp_eq(x - y, (wa + (GFp::MOD as u128) * 2) - wb);
        check_gfp_eq(-y, (GFp::MOD as u128) * 2 - wb);
        check_gfp_eq(x * y, wa * wb);
        check_gfp_eq(x.mul_small(small_b as u32), wa * small_b);
        check_gfp_eq(x.square(), wa * wa);
        if a == 0 || a == GFp::MOD {
            check_gfp_eq(x.invert(), 0);
        } else {
            check_gfp_eq(x * x.invert(), 1);
        }
        assert!(x.double().equals(&(x + x)) == 0xFFFFFFFF);
        assert!(x.half().double().equals(&x) == 0xFFFFFFFF);
        assert!((x - y).equals(&(y - x).neg()) == 0xFFFFFFFF);
    }

    #[test]
    fn gfp_ops() {
        for i in 0..10 {
            let v: u64 = (i as u64) + GFp::MOD - 5;
            if i <= 4 {
                let (x, c) = GFp::from_u64(v);
                assert!(c == 0xFFFFFFFFFFFFFFFF);
                assert!(x.to_u64() == v);
                let y = GFp::from_u64_reduce(v);
                assert!(y.to_u64() == v);
            } else {
                let v2 = v - GFp::MOD;
                let (x, c) = GFp::from_u64(v);
                assert!(c == 0);
                assert!(x.to_u64() == 0);
                let y = GFp::from_u64_reduce(v);
                assert!(y.to_u64() == v2);
            }
        }

        test_gfp_ops(0, 0);
        test_gfp_ops(0, 1);
        test_gfp_ops(1, 0);
        test_gfp_ops(1, 1);
        test_gfp_ops(0, 0xFFFFFFFFFFFFFFFF);
        test_gfp_ops(0xFFFFFFFFFFFFFFFF, 0);
        test_gfp_ops(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
        test_gfp_ops(0, 0xFFFFFFFF00000000);
        test_gfp_ops(0xFFFFFFFF00000000, 0);
        test_gfp_ops(0xFFFFFFFF00000000, 0xFFFFFFFF00000000);
        let mut prng = PRNG(0);
        for _ in 0..10000 {
            let a = prng.next_u64();
            let b = prng.next_u64();
            test_gfp_ops(a, b);
        }
        assert!(GFp::ZERO.legendre() == 0);
        let (s0, c0) = GFp::ZERO.sqrt();
        check_gfp_eq(s0, 0);
        assert!(c0 == 0xFFFFFFFF);
        for _ in 0..1000 {
            let x = GFp::from_u64_reduce((prng.next_u64() >> 1) + 1).square();
            assert!(x.legendre() == 1);
            let (r1, c1) = x.sqrt();
            assert!(r1.square().equals(&x) == 0xFFFFFFFF);
            assert!(c1 == 0xFFFFFFFF);
            let y = x * GFp::from_u64_reduce(7);
            assert!(y.legendre() == -1);
            let (r2, c2) = y.sqrt();
            assert!(r2.iszero() == 0xFFFFFFFF);
            assert!(c2 == 0);

            let z = x.square();
            let (r3, c3) = z.fourth_root();
            assert!(r3.xsquare(2).equals(&z) == 0xFFFFFFFF);
            assert!(c3 == 0xFFFFFFFF);

            let z4 = z.square();
            let (r3, c3) = z4.eighth_root();
            assert!(r3.xsquare(3).equals(&z4) == 0xFFFFFFFF);
            assert!(c3 == 0xFFFFFFFF);
        }
    }

    #[test]
    fn test_batch_invert_6() {
        let mut prng = PRNG(0);
        for _ in 0..100 {
            let mut inputs = [GFp::ZERO; 6];
            for i in 0..6 {
                inputs[i] = GFp::from_u64_reduce(prng.next_u64());
            }
            // Introduce some zeros explicitly
            if prng.next_u64() % 4 == 0 {
                inputs[2] = GFp::ZERO;
            }
            if prng.next_u64() % 4 == 0 {
                inputs[5] = GFp::ZERO;
            }

            let batch_res = GFp::batch_invert_fixed::<6>(&inputs);
            for i in 0..6 {
                let single_res = inputs[i].invert();
                assert!(batch_res[i].equals(&single_res) == 0xFFFFFFFF);
            }
        }
    }

    #[test]
    fn test_batch_sqrt() {
        let mut prng = PRNG(0);
        for _ in 0..100 {
            let mut inputs = [GFp::ZERO; 6];
            for i in 0..6 {
                inputs[i] = GFp::from_u64_reduce(prng.next_u64()).square();
            }
            inputs[2] = GFp::ZERO;

            let results = GFp::batch_sqrt::<6>(&inputs);
            for i in 0..6 {
                let (expected_res, expected_r) = inputs[i].sqrt();
                let (actual_res, actual_r) = results[i];
                assert_eq!(expected_r, actual_r);
                assert!(actual_res.equals(&expected_res) == 0xFFFFFFFF);
            }
        }
    }
}
