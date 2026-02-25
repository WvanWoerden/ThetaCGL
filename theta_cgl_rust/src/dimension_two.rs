#![allow(non_snake_case)]

// Macro expectations:
// Fq      type of field element Fp^2
// X0, Z0, U0, V0, type Fq, coordinates of theta null point
macro_rules! define_dim_two_theta_core {
    () => {
        use crate::util::pad_msg;

        // Theta Point
        // The domain / codomain is described by a theta point
        #[derive(Clone, Copy, Debug)]
        pub struct ThetaPointDim2 {
            pub X: Fq,
            pub Z: Fq,
            pub U: Fq,
            pub V: Fq,
        }

        impl ThetaPointDim2 {
            pub const fn new(X: &Fq, Z: &Fq, U: &Fq, V: &Fq) -> ThetaPointDim2 {
                Self {
                    X: *X,
                    Z: *Z,
                    U: *U,
                    V: *V,
                }
            }

            pub fn coords(self) -> (Fq, Fq, Fq, Fq) {
                (self.X, self.Z, self.U, self.V)
            }

            // Compute the Hadamard transform
            // Cost 8 additions
            fn to_hadamard(self, X: &Fq, Z: &Fq, U: &Fq, V: &Fq) -> (Fq, Fq, Fq, Fq) {
                let t1 = X + Z;
                let t2 = X - Z;
                let t3 = U + V;
                let t4 = U - V;

                (&t1 + &t3, &t2 + &t4, &t1 - &t3, &t2 - &t4)
            }

            // Squared theta first squares the coords of the points
            // and then computes the Hadamard transformation.
            // This gives the square of the dual coords
            // Cost 4S + 8a
            pub fn squared_theta(self) -> (Fq, Fq, Fq, Fq) {
                let XX = self.X.square();
                let ZZ = self.Z.square();
                let UU = self.U.square();
                let VV = self.V.square();

                self.to_hadamard(&XX, &ZZ, &UU, &VV)
            }

            // Compute the two isogeny
            pub fn radical_two_isogeny(self, bits: Vec<u8>) -> ThetaPointDim2 {
                // Compute squared dual coordinates xi
                let (x0, x1, x2, x3) = self.squared_theta();

                // Compute yi = sqrt(x0 * xi)
                let inputs = [&x0 * &x1, &x0 * &x2, &x0 * &x3];
                let results = Fq::batch_sqrt::<3>(&inputs);
                let mut y1 = results[0].0;
                let mut y2 = results[1].0;
                let mut y3 = results[2].0;

                // Consume bits by setting signs
                let ctl1 = ((bits[0] as u32) & 1).wrapping_neg();
                y1.set_condneg(ctl1);

                let ctl2 = ((bits[1] as u32) & 1).wrapping_neg();
                y2.set_condneg(ctl2);

                let ctl3 = ((bits[2] as u32) & 1).wrapping_neg();
                y3.set_condneg(ctl3);

                // Compute the codomain
                let (b0, b1, b2, b3) = self.to_hadamard(&x0, &y1, &y2, &y3);

                ThetaPointDim2::new(&b0, &b1, &b2, &b3)
            }

            // Compute a four-radical isogeny
            pub fn radical_four_isogeny(self, bits: Vec<u8>) -> ThetaPointDim2 {
                // Compute squared dual coordinates xi
                let (x0, x1, x2, x3) = self.squared_theta();

                let x01 = &x0 * &x1;
                let x02 = &x0 * &x2;
                let x13 = &x1 * &x3;
                let x23 = &x2 * &x3;

                // Compute y and flip the sign if bit[0] is set to 1
                let mut y = (&x01 * &x23).sqrt().0;
                let ctl1 = ((bits[0] as u32) & 1).wrapping_neg();
                y.set_condneg(ctl1);

                // Compute alpha_1 and alpha2
                let alpha_1_4 = (&y.mul2() + &x01 + &x23).mul4();
                let alpha_2_4 = (&y.mul2() + &x02 + &x13).mul4();
                let mut alpha_1 = alpha_1_4.fourth_root().0;
                let mut alpha_2 = alpha_2_4.fourth_root().0;

                // Multiply alpha_1 by a fourth root of unity
                // depending on bits 1 and 2
                let ctl2 = ((bits[1] as u32) & 1).wrapping_neg();
                let ctl3 = ((bits[2] as u32) & 1).wrapping_neg();
                alpha_1.set_cond(&(&alpha_1 * &Fq::ZETA), ctl3);
                alpha_1.set_condneg(ctl2);

                // Multiply alpha_2 by a fourth root of unity
                // depending on bits 3 and 4
                let ctl4 = ((bits[3] as u32) & 1).wrapping_neg();
                let ctl5 = ((bits[4] as u32) & 1).wrapping_neg();
                alpha_2.set_cond(&(&alpha_2 * &Fq::ZETA), ctl5);
                alpha_2.set_condneg(ctl4);

                let mut alpha_3_2 = (&x23 + &y).mul8();
                alpha_3_2 *= (&x02 + &y) * &x23 * &x3 + &(&x13 + &y) * &x23 * &x2;
                let mut alpha_3 = alpha_3_2.sqrt().0;

                // Change the sign of alpha_3 depending on bit 5
                let ctl6 = ((bits[5] as u32) & 1).wrapping_neg();
                alpha_3.set_condneg(ctl6);

                // Projective factor
                let lambda = &x23 * &alpha_1 * &alpha_2;

                let (b0, b1, b2, b3) = self.to_hadamard(
                    &(&self.X.mul2() * &lambda),
                    &(&alpha_1 * &lambda),
                    &(&alpha_2 * &lambda),
                    &alpha_3,
                );

                ThetaPointDim2::new(&b0, &b1, &b2, &b3)
            }

            pub fn to_hash(self) -> (Fq, Fq, Fq) {
                let (X, Z, U, V) = (self.X, self.Z, self.U, self.V);
                let X_inv = X.invert();

                (&Z * &X_inv, &U * &X_inv, &V * &X_inv)
            }
        }

        #[derive(Clone, Copy, Debug)]
        pub struct CGLDim2Rad2 {
            chunk_len: usize,
            block_size: usize,
        }

        impl CGLDim2Rad2 {
            const O0: ThetaPointDim2 = ThetaPointDim2::new(&X0, &Z0, &U0, &V0);

            pub fn new(block_size: usize) -> Self {
                let chunk_len = 3;
                assert!(block_size % chunk_len == 0);
                Self {
                    chunk_len,
                    block_size,
                }
            }
            pub fn bit_string(self, mut T: ThetaPointDim2, msg: Vec<u8>) -> ThetaPointDim2 {
                let iter = msg.chunks(self.chunk_len);
                for i in iter {
                    T = T.radical_two_isogeny(i.to_vec());
                }

                T
            }

            pub fn hash(self, msg: Vec<u8>) -> (Fq, Fq, Fq) {
                let padded_msg = pad_msg(msg, self.block_size);
                let T = self.bit_string(Self::O0, padded_msg);
                T.to_hash()
            }
        }

        #[derive(Clone, Copy, Debug)]
        pub struct CGLDim2Rad4 {
            chunk_len: usize,
            block_size: usize,
        }

        impl CGLDim2Rad4 {
            const O0: ThetaPointDim2 = ThetaPointDim2::new(&X0, &Z0, &U0, &V0);

            pub fn new(block_size: usize) -> Self {
                let chunk_len = 6;
                assert!(block_size % chunk_len == 0);
                Self {
                    chunk_len,
                    block_size,
                }
            }

            pub fn bit_string(self, mut T: ThetaPointDim2, msg: Vec<u8>) -> ThetaPointDim2 {
                let iter = msg.chunks(self.chunk_len);
                for i in iter {
                    T = T.radical_four_isogeny(i.to_vec());
                }

                T
            }

            pub fn hash(&self, msg: Vec<u8>) -> (Fq, Fq, Fq) {
                let padded_msg = pad_msg(msg, self.block_size);
                let T = self.bit_string(Self::O0, padded_msg);
                T.to_hash()
            }
        }
    };
} // End of macro: define_dim_two_theta_core

pub(crate) use define_dim_two_theta_core;
