pub struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, index: 624 }
    }

    pub fn py_seed(seed: u32) -> Self {
        let mut rng = Mt19937 {
            mt: [0u32; 624],
            index: 624,
        };
        rng.init_by_array(&[seed]);
        rng
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            self.mt[i] = 1812433253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    pub fn init_by_array(&mut self, key: &[u32]) {
        const N: usize = 624;
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = N.max(key.len());
        while k > 0 {
            let t = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            self.mt[i] = t;
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let t = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941)))
            .wrapping_sub(i as u32);
            self.mt[i] = t;
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x80000000;
        self.index = N;
    }

    fn genrand(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER: u32 = 0x80000000;
        const LOWER: u32 = 0x7fffffff;
        if self.index >= N {
            let mut kk = 0;
            while kk < N - M {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] =
                    self.mt[kk + M] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
                kk += 1;
            }
            while kk < N - 1 {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] =
                    self.mt[kk + M - N] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
                kk += 1;
            }
            let y = (self.mt[N - 1] & UPPER) | (self.mt[0] & LOWER);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k <= 32 {
            self.genrand() >> (32 - k)
        } else {
            let mut result: u64 = 0;
            let words = (k - 1) / 32 + 1;
            for i in 0..words {
                let mut r = self.genrand();
                if i == words - 1 {
                    r >>= 32 * words - k;
                }
                result = (result << 32) | r as u64;
            }
            result as u32
        }
    }

    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let mut i = items.len();
        while i > 1 {
            let j = self.randbelow(i as u32) as usize;
            items.swap(i - 1, j);
            i -= 1;
        }
    }
}
