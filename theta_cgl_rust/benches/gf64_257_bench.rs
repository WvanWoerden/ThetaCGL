use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand_core::{CryptoRng, Error, RngCore};
use theta_cgl_rust::finitefield::gf64_257::GFp;

// Simple PRNG to avoid pulling in full rand crate if not needed
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RngCore for SimpleRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let v = self.next_u64();
            let bytes = v.to_le_bytes();
            let n = std::cmp::min(dest.len() - i, 8);
            dest[i..i + n].copy_from_slice(&bytes[..n]);
            i += n;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for SimpleRng {}

fn bench_sqrt(c: &mut Criterion) {
    let mut rng = SimpleRng::new(12345);
    let x = GFp::rand(&mut rng);

    c.bench_function("GFp::sqrt", |b| b.iter(|| black_box(x).sqrt()));

    let inputs = [
        GFp::rand(&mut rng),
        GFp::rand(&mut rng),
        GFp::rand(&mut rng),
        GFp::rand(&mut rng),
        GFp::rand(&mut rng),
        GFp::rand(&mut rng),
    ];
    c.bench_function("GFp::sqrt x6 (serial)", |b| {
        b.iter(|| {
            let mut out = [(GFp::ZERO, 0); 6];
            for i in 0..6 {
                out[i] = black_box(inputs[i]).sqrt();
            }
            out
        })
    });

    c.bench_function("GFp::batch_sqrt::<6>", |b| {
        b.iter(|| GFp::batch_sqrt::<6>(black_box(&inputs)))
    });
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn measure_sqrt_cycles() {
    use core::arch::x86_64::_rdtsc;
    let mut rng = SimpleRng::new(12345);
    let x = GFp::rand(&mut rng);

    // Warmup
    for _ in 0..1000 {
        black_box(x).sqrt();
    }

    let iterations = 10000;
    let start = unsafe { _rdtsc() };
    for _ in 0..iterations {
        black_box(x).sqrt();
    }
    let end = unsafe { _rdtsc() };

    let total_cycles = end - start;
    let avg_cycles = total_cycles as f64 / iterations as f64;
    println!("GFp::sqrt average cycles: {:.2}", avg_cycles);

    // Measure batch cycles
    let inputs = [GFp::rand(&mut rng); 6];
    // Warmup
    for _ in 0..1000 {
        GFp::batch_sqrt::<6>(black_box(&inputs));
    }

    let start_batch = unsafe { _rdtsc() };
    for _ in 0..iterations {
        GFp::batch_sqrt::<6>(black_box(&inputs));
    }
    let end_batch = unsafe { _rdtsc() };

    let total_cycles_batch = end_batch - start_batch;
    let avg_cycles_batch = total_cycles_batch as f64 / iterations as f64;
    println!(
        "GFp::batch_sqrt::<6> average cycles (total for 6): {:.2}",
        avg_cycles_batch
    );
    println!(
        "GFp::batch_sqrt::<6> average cycles (per element): {:.2}",
        avg_cycles_batch / 6.0
    );
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn measure_sqrt_cycles() {
    println!("Cycle counting not supported on this architecture.");
}

criterion_group!(benches, bench_sqrt);

fn main() {
    measure_sqrt_cycles();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
