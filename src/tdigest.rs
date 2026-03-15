//! Backend originally by Paul Meng (https://github.com/MnO2/t-digest)

use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::TryReserveError;

pub const TD_SIZE_DEFAULT: usize = 1000;
pub const TD_SIZE_PLATFORM_MAX: usize = (isize::MAX / 16) as usize;
pub const TD_SIZE_GLOBAL_MAX: usize = (i64::MAX / 16) as usize;

#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "use_serde", derive(Serialize, Deserialize))]
pub struct Centroid {
    pub mean: OrderedFloat<f64>,
    pub weight: OrderedFloat<f64>,
}

impl PartialOrd for Centroid {
    fn partial_cmp(&self, other: &Centroid) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Centroid {
    fn cmp(&self, other: &Centroid) -> Ordering {
        self.mean.cmp(&other.mean)
    }
}

impl Centroid {
    pub fn new(mean: f64, weight: f64) -> Self {
        Centroid {
            mean: OrderedFloat::from(mean),
            weight: OrderedFloat::from(weight),
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean.into_inner()
    }

    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight.into_inner()
    }

    pub fn add(&mut self, sum: f64, weight: f64) -> f64 {
        let weight_: f64 = self.weight.into_inner();
        let mean_: f64 = self.mean.into_inner();

        let new_sum: f64 = sum + weight_ * mean_;
        let new_weight: f64 = weight_ + weight;
        self.weight = OrderedFloat::from(new_weight);
        self.mean = OrderedFloat::from(new_sum / new_weight);
        new_sum
    }
}

impl Default for Centroid {
    fn default() -> Self {
        Centroid {
            mean: OrderedFloat::from(0.0),
            weight: OrderedFloat::from(1.0),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "use_serde", derive(Serialize, Deserialize))]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    mass: OrderedFloat<f64>,
    sum: OrderedFloat<f64>,
    min: OrderedFloat<f64>,
    max: OrderedFloat<f64>,
    count: u128,
}

impl TDigest {
    const MAGIC: [u8; 8] = *b"FASTDIG~";
    const VERSION: u32 = 1;
    const HEADER_BYTES: usize = 80; // beginning of centroids in binary format
    const PADDING_BYTES: usize = 4; // HEADER_BYTES - sum(used header bytes)
    const TARGET_DIGITS: u32 = 8;
    const RECOMP_THRESH: u128 = 10u128.pow(f64::DIGITS - Self::TARGET_DIGITS);

    pub fn new_with_size(max_size: usize) -> Result<Self, TryReserveError> {
        let mut centroids: Vec<Centroid> = Vec::new();
        centroids.try_reserve_exact(max_size)?;

        Ok(TDigest {
            centroids,
            max_size,
            mass: OrderedFloat::from(0.0),
            sum: OrderedFloat::from(0.0),
            min: OrderedFloat::from(f64::NAN),
            max: OrderedFloat::from(f64::NAN),
            count: 0,
        })
    }

    pub fn new(
        centroids: Vec<Centroid>,
        max_size: usize,
        mass: f64,
        sum: f64,
        min: f64,
        max: f64,
        count: u128,
    ) -> Result<Self, TryReserveError> {
        if centroids.len() <= max_size {
            Ok(TDigest {
                centroids,
                max_size,
                mass: OrderedFloat::from(mass),
                sum: OrderedFloat::from(sum),
                min: OrderedFloat::from(min),
                max: OrderedFloat::from(max),
                count,
            })
        } else {
            let sz = centroids.len();
            let digests: Vec<TDigest> = vec![
                TDigest::new_with_size(max_size)?,
                TDigest::new(centroids, sz, mass, sum, min, max, count)?,
            ];
            Self::merge_digests(digests, Some(max_size))
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BytesError> {
        #[inline]
        fn read<const N: usize>(bytes: &[u8], offset: &mut usize) -> [u8; N] {
            let mut out = [0u8; N];
            out.copy_from_slice(&bytes[*offset..*offset + N]);
            *offset += N;
            out
        }

        fn validate_u64_size(value: u64) -> Result<usize, BytesError> {
            match value {
                n if n > TD_SIZE_GLOBAL_MAX as u64 => {
                    Err(BytesError::CorruptData)
                }
                n if n > TD_SIZE_PLATFORM_MAX as u64 => {
                    Err(BytesError::WrongArch)
                }
                n => Ok(n as usize),
            }
        }

        let mut offset: usize = 0;

        if bytes.is_empty() {
            return Err(BytesError::EmptyData);
        }

        if bytes.len() < 12 || read::<8>(bytes, &mut offset) != Self::MAGIC {
            return Err(BytesError::WrongFormat);
        }

        let version = u32::from_le_bytes(read::<4>(bytes, &mut offset));
        if version != Self::VERSION {
            return Err(BytesError::WrongVersion);
        }

        if bytes.len() < Self::HEADER_BYTES {
            return Err(BytesError::CorruptData);
        }

        let c_len_u64 = u64::from_le_bytes(read::<8>(bytes, &mut offset));
        let centroids_len = validate_u64_size(c_len_u64)?;

        let expected = Self::HEADER_BYTES + centroids_len * 16;
        if bytes.len() != expected {
            return Err(BytesError::CorruptData);
        }

        let max_size_u64 = u64::from_le_bytes(read::<8>(bytes, &mut offset));
        let max_size = validate_u64_size(max_size_u64)?;

        let mass = f64::from_le_bytes(read::<8>(bytes, &mut offset));
        let sum = f64::from_le_bytes(read::<8>(bytes, &mut offset));
        let min = f64::from_le_bytes(read::<8>(bytes, &mut offset));
        let max = f64::from_le_bytes(read::<8>(bytes, &mut offset));
        let count = u128::from_le_bytes(read::<16>(bytes, &mut offset));

        offset = Self::HEADER_BYTES;

        let mut centroids: Vec<Centroid> = Vec::new();
        centroids
            .try_reserve_exact(centroids_len)
            .map_err(BytesError::MemError)?;

        for _ in 0..centroids_len {
            let mean = f64::from_le_bytes(read::<8>(bytes, &mut offset));
            let weight = f64::from_le_bytes(read::<8>(bytes, &mut offset));
            centroids.push(Centroid::new(mean, weight));
        }

        Ok(Self {
            centroids,
            max_size,
            mass: OrderedFloat::from(mass),
            sum: OrderedFloat::from(sum),
            min: OrderedFloat::from(min),
            max: OrderedFloat::from(max),
            count,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, TryReserveError> {
        let centroids_len = self.centroids.len();
        let cap = Self::HEADER_BYTES + centroids_len * 16;
        let mut buf: Vec<u8> = Vec::new();
        buf.try_reserve_exact(cap)?;

        buf.extend_from_slice(&Self::MAGIC);
        buf.extend_from_slice(&Self::VERSION.to_le_bytes());
        buf.extend_from_slice(&(centroids_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.max_size as u64).to_le_bytes());
        buf.extend_from_slice(&self.mass.into_inner().to_le_bytes());
        buf.extend_from_slice(&self.sum.into_inner().to_le_bytes());
        buf.extend_from_slice(&self.min.into_inner().to_le_bytes());
        buf.extend_from_slice(&self.max.into_inner().to_le_bytes());
        buf.extend_from_slice(&self.count.to_le_bytes());
        buf.extend_from_slice(&[0u8; Self::PADDING_BYTES]);

        for c in &self.centroids {
            buf.extend_from_slice(&c.mean().to_le_bytes());
            buf.extend_from_slice(&c.weight().to_le_bytes());
        }
        Ok(buf)
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.sum() / self.mass()
    }

    #[inline]
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    #[inline]
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size
    }

    #[inline]
    pub fn mass(&self) -> f64 {
        self.mass.into_inner()
    }

    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum.into_inner()
    }

    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner()
    }

    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner()
    }

    #[inline]
    pub fn count(&self) -> u128 {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }
}

impl Default for TDigest {
    fn default() -> Self {
        TDigest::new_with_size(TD_SIZE_DEFAULT)
            .expect("default max size should be allocatable")
    }
}

impl TDigest {
    fn k_to_q(k: f64, d: f64) -> f64 {
        let k_div_d = k / d;
        if k_div_d >= 0.5 {
            let base = 1.0 - k_div_d;
            1.0 - 2.0 * base * base
        } else {
            2.0 * k_div_d * k_div_d
        }
    }

    pub fn merge_unsorted(
        &self,
        unsorted_values: Vec<f64>,
    ) -> Result<TDigest, TryReserveError> {
        if unsorted_values.is_empty() {
            return Ok(self.clone());
        }

        let mut sorted_values: Vec<OrderedFloat<f64>> = unsorted_values
            .into_iter()
            .map(OrderedFloat::from)
            .collect();
        sorted_values.sort();

        self.merge_sorted(sorted_values)
    }

    pub fn merge_unsorted_weighted(
        &self,
        unsorted_values: Vec<f64>,
        unsorted_weights: Vec<f64>,
    ) -> Result<TDigest, TryReserveError> {
        debug_assert_eq!(unsorted_values.len(), unsorted_weights.len());
        if unsorted_values.is_empty() {
            return Ok(self.clone());
        }

        let mut pairs: Vec<(OrderedFloat<f64>, f64)> = unsorted_values
            .into_iter()
            .zip(unsorted_weights)
            .map(|(value, weight)| (OrderedFloat::from(value), weight))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        self.merge_sorted_weighted(pairs)
    }

    pub fn merge_sorted(
        &self,
        sorted_values: Vec<OrderedFloat<f64>>,
    ) -> Result<TDigest, TryReserveError> {
        if sorted_values.is_empty() {
            return Ok(self.clone());
        }

        let mut result = TDigest::new_with_size(self.max_size)?;
        result.count = self.count + sorted_values.len() as u128;
        result.mass =
            OrderedFloat::from(self.mass() + (sorted_values.len() as f64));

        let maybe_min = *sorted_values.first().unwrap();
        let maybe_max = *sorted_values.last().unwrap();

        if self.mass() > 0.0 {
            result.min = std::cmp::min(self.min, maybe_min);
            result.max = std::cmp::max(self.max, maybe_max);
        } else {
            result.min = maybe_min;
            result.max = maybe_max;
        }

        let mut compressed: Vec<Centroid> = Vec::new();
        compressed.try_reserve_exact(self.max_size)?;

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_mass: f64 =
            Self::k_to_q(k_limit, self.max_size as f64) * result.mass();
        k_limit += 1.0;

        let mut iter_centroids = self.centroids.iter().peekable();
        let mut iter_sorted_values = sorted_values.iter().peekable();

        let mut curr: Centroid = if let Some(c) = iter_centroids.peek() {
            if c.mean() < iter_sorted_values.peek().unwrap().into_inner() {
                iter_centroids.next().unwrap().clone()
            } else {
                Centroid::new(
                    iter_sorted_values.next().unwrap().into_inner(),
                    1.0,
                )
            }
        } else {
            Centroid::new(iter_sorted_values.next().unwrap().into_inner(), 1.0)
        };

        let mut weight_so_far: f64 = curr.weight();
        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        while iter_centroids.peek().is_some()
            || iter_sorted_values.peek().is_some()
        {
            let next: Centroid = if let Some(c) = iter_centroids.peek() {
                if iter_sorted_values.peek().is_none()
                    || c.mean()
                        < iter_sorted_values.peek().unwrap().into_inner()
                {
                    iter_centroids.next().unwrap().clone()
                } else {
                    Centroid::new(
                        iter_sorted_values.next().unwrap().into_inner(),
                        1.0,
                    )
                }
            } else {
                Centroid::new(
                    iter_sorted_values.next().unwrap().into_inner(),
                    1.0,
                )
            };

            let next_sum: f64 = next.mean() * next.weight();
            weight_so_far += next.weight();

            if weight_so_far <= q_limit_times_mass {
                sums_to_merge += next_sum;
                weights_to_merge += next.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum() + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;

                compressed.push(curr.clone());
                q_limit_times_mass =
                    Self::k_to_q(k_limit, self.max_size as f64) * result.mass();
                k_limit += 1.0;
                curr = next;
            }
        }

        result.sum = OrderedFloat::from(
            result.sum() + curr.add(sums_to_merge, weights_to_merge),
        );
        compressed.push(curr);
        compressed.shrink_to_fit();
        compressed.sort();

        result.centroids = compressed;
        result.maybe_recompute_totals(self.count);

        Ok(result)
    }

    pub fn merge_sorted_weighted(
        &self,
        sorted_values_weights: Vec<(OrderedFloat<f64>, f64)>,
    ) -> Result<TDigest, TryReserveError> {
        if sorted_values_weights.is_empty() {
            return Ok(self.clone());
        }

        let total_new_weight: f64 = sorted_values_weights
            .iter()
            .map(|(_, weight)| *weight)
            .sum();

        let mut result = TDigest::new_with_size(self.max_size)?;
        result.count = self.count + sorted_values_weights.len() as u128;
        result.mass = OrderedFloat::from(self.mass() + total_new_weight);

        let maybe_min = sorted_values_weights.first().unwrap().0;
        let maybe_max = sorted_values_weights.last().unwrap().0;

        if self.mass() > 0.0 {
            result.min = std::cmp::min(self.min, maybe_min);
            result.max = std::cmp::max(self.max, maybe_max);
        } else {
            result.min = maybe_min;
            result.max = maybe_max;
        }

        let mut compressed: Vec<Centroid> = Vec::new();
        compressed.try_reserve_exact(self.max_size)?;

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_mass: f64 =
            Self::k_to_q(k_limit, self.max_size as f64) * result.mass();
        k_limit += 1.0;

        let mut iter_centroids = self.centroids.iter().peekable();
        let mut iter_values_weights = sorted_values_weights.iter().peekable();

        let mut curr: Centroid = if let Some(c) = iter_centroids.peek() {
            if c.mean() < iter_values_weights.peek().unwrap().0.into_inner() {
                iter_centroids.next().unwrap().clone()
            } else {
                let (val, weight) = *iter_values_weights.next().unwrap();
                Centroid::new(val.into_inner(), weight)
            }
        } else {
            let (val, weight) = *iter_values_weights.next().unwrap();
            Centroid::new(val.into_inner(), weight)
        };

        let mut weight_so_far: f64 = curr.weight();
        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        while iter_centroids.peek().is_some()
            || iter_values_weights.peek().is_some()
        {
            let next: Centroid = if let Some(c) = iter_centroids.peek() {
                if iter_values_weights.peek().is_none()
                    || c.mean()
                        < iter_values_weights.peek().unwrap().0.into_inner()
                {
                    iter_centroids.next().unwrap().clone()
                } else {
                    let (val, weight) = *iter_values_weights.next().unwrap();
                    Centroid::new(val.into_inner(), weight)
                }
            } else {
                let (val, weight) = *iter_values_weights.next().unwrap();
                Centroid::new(val.into_inner(), weight)
            };

            let next_sum: f64 = next.mean() * next.weight();
            weight_so_far += next.weight();

            if weight_so_far <= q_limit_times_mass {
                sums_to_merge += next_sum;
                weights_to_merge += next.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum() + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;

                compressed.push(curr.clone());
                q_limit_times_mass =
                    Self::k_to_q(k_limit, self.max_size as f64) * result.mass();
                k_limit += 1.0;
                curr = next;
            }
        }

        result.sum = OrderedFloat::from(
            result.sum() + curr.add(sums_to_merge, weights_to_merge),
        );
        compressed.push(curr);
        compressed.shrink_to_fit();
        compressed.sort();

        result.centroids = compressed;
        result.maybe_recompute_totals(self.count);

        Ok(result)
    }

    fn external_merge(
        centroids: &mut [Centroid],
        first: usize,
        middle: usize,
        last: usize,
    ) -> Result<(), TryReserveError> {
        let mut result: Vec<Centroid> = Vec::new();
        result.try_reserve_exact(centroids.len())?;

        let mut i = first;
        let mut j = middle;

        while i < middle && j < last {
            match centroids[i].cmp(&centroids[j]) {
                Ordering::Less => {
                    result.push(centroids[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    result.push(centroids[j].clone());
                    j += 1;
                }
                Ordering::Equal => {
                    result.push(centroids[i].clone());
                    i += 1;
                }
            }
        }

        while i < middle {
            result.push(centroids[i].clone());
            i += 1;
        }

        while j < last {
            result.push(centroids[j].clone());
            j += 1;
        }

        i = first;
        for centroid in result.into_iter() {
            centroids[i] = centroid;
            i += 1;
        }

        Ok(())
    }

    pub fn merge_digests(
        digests: Vec<TDigest>,
        max_size: Option<usize>,
    ) -> Result<TDigest, TryReserveError> {
        let max_size = if let Some(max) = max_size {
            max
        } else {
            digests
                .iter()
                .map(|digest| digest.max_size)
                .max()
                .unwrap_or(TD_SIZE_DEFAULT)
        };

        let n_centroids: usize =
            digests.iter().map(|d| d.centroids.len()).sum();
        if n_centroids == 0 {
            return TDigest::new_with_size(max_size);
        }

        let mut centroids: Vec<Centroid> = Vec::new();
        centroids.try_reserve_exact(n_centroids)?;
        let mut starts: Vec<usize> = Vec::new();
        starts.try_reserve_exact(digests.len())?;

        let count: u128 = digests.iter().map(|d| d.count).sum();
        let max_count: u128 = digests.iter().map(|d| d.count).max().unwrap();

        let mut mass: f64 = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        let mut start: usize = 0;
        for digest in digests.into_iter() {
            starts.push(start);

            let curr_mass: f64 = digest.mass();
            if curr_mass > 0.0 {
                min = std::cmp::min(min, digest.min);
                max = std::cmp::max(max, digest.max);
                mass += curr_mass;
                for centroid in digest.centroids {
                    centroids.push(centroid);
                    start += 1;
                }
            }
        }

        let mut digests_per_block: usize = 1;
        while digests_per_block < starts.len() {
            for i in (0..starts.len()).step_by(digests_per_block * 2) {
                if i + digests_per_block < starts.len() {
                    let first = starts[i];
                    let middle = starts[i + digests_per_block];
                    let last = if i + 2 * digests_per_block < starts.len() {
                        starts[i + 2 * digests_per_block]
                    } else {
                        centroids.len()
                    };

                    debug_assert!(first <= middle && middle <= last);
                    Self::external_merge(&mut centroids, first, middle, last)?;
                }
            }

            digests_per_block *= 2;
        }

        let mut result = TDigest::new_with_size(max_size)?;
        let mut compressed: Vec<Centroid> = Vec::new();
        compressed.try_reserve_exact(max_size)?;

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_mass: f64 =
            Self::k_to_q(k_limit, max_size as f64) * mass;

        let mut iter_centroids = centroids.iter_mut();
        let mut curr = iter_centroids.next().unwrap();
        let mut weight_so_far: f64 = curr.weight();
        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        for centroid in iter_centroids {
            weight_so_far += centroid.weight();

            if weight_so_far <= q_limit_times_mass {
                sums_to_merge += centroid.mean() * centroid.weight();
                weights_to_merge += centroid.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum() + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;
                compressed.push(curr.clone());
                q_limit_times_mass =
                    Self::k_to_q(k_limit, max_size as f64) * mass;
                k_limit += 1.0;
                curr = centroid;
            }
        }

        result.sum = OrderedFloat::from(
            result.sum() + curr.add(sums_to_merge, weights_to_merge),
        );
        compressed.push(curr.clone());
        compressed.shrink_to_fit();
        compressed.sort();

        result.centroids = compressed;
        result.mass = OrderedFloat::from(mass);
        result.min = min;
        result.max = max;
        result.count = count;

        result.maybe_recompute_totals(max_count);

        Ok(result)
    }

    /// Function by Andy Lok (https://github.com/andylokandy/tdigests)
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        if self.centroids.len() == 1 {
            return self.centroids[0].mean();
        }

        let mut cumulative = 0.0;
        let mut cum_left = 0.0;
        let mut cum_right = 0.0;
        let mut position = 0;

        for (k, centroid) in self.centroids.iter().enumerate() {
            cum_left = cum_right;
            cum_right = (2.0 * cumulative + centroid.weight() - 1.0)
                / 2.0
                / (self.mass() - 1.0);
            cumulative += centroid.weight();

            if cum_right >= q {
                break;
            }

            position = k + 1;
        }

        if position == 0 {
            return self.centroids[0].mean();
        }

        if position >= self.centroids.len() {
            return self.centroids[self.centroids.len() - 1].mean();
        }

        let centroid_left = &self.centroids[position - 1];
        let centroid_right = &self.centroids[position];

        let weight_between = cum_right - cum_left;
        let fraction = (q - cum_left) / weight_between;

        centroid_left.mean() * (1.0 - fraction)
            + centroid_right.mean() * fraction
    }

    pub fn estimate_quantiles(
        &self,
        qs: &[f64],
    ) -> Result<Vec<f64>, TryReserveError> {
        let n_centroids = self.centroids.len();

        if n_centroids == 0 {
            return Ok(vec![]);
        }

        if n_centroids == 1 {
            let m = self.centroids[0].mean();
            return Ok(qs.iter().map(|_| m).collect());
        }

        let mut cum_left: Vec<f64> = Vec::new();
        let mut cum_right: Vec<f64> = Vec::new();
        cum_left.try_reserve_exact(n_centroids)?;
        cum_right.try_reserve_exact(n_centroids)?;

        let mut cumulative = 0.0;
        let mut prev_right = 0.0;

        for centroid in &self.centroids {
            let left = prev_right;
            let right = (2.0 * cumulative + centroid.weight() - 1.0)
                / 2.0
                / (self.mass() - 1.0);
            cumulative += centroid.weight();
            prev_right = right;
            cum_left.push(left);
            cum_right.push(right);
        }

        let means: Vec<f64> = self.centroids.iter().map(|c| c.mean()).collect();

        let mut out: Vec<f64> = Vec::new();
        out.try_reserve_exact(qs.len())?;

        for &q in qs {
            let idx = cum_right
                .binary_search_by(|x| x.partial_cmp(&q).unwrap())
                .unwrap_or_else(|i| i);

            if idx == 0 {
                out.push(means[0]);
                continue;
            }

            if idx >= n_centroids {
                out.push(means[n_centroids - 1]);
                continue;
            }

            let left = cum_left[idx];
            let right = cum_right[idx];
            let weight_between = right - left;

            if weight_between == 0.0 {
                out.push(means[idx]);
                continue;
            }

            let fraction = (q - left) / weight_between;
            let m_left = means[idx - 1];
            let m_right = means[idx];
            out.push(m_left * (1.0 - fraction) + m_right * fraction);
        }
        Ok(out)
    }

    /// Function by Andy Lok (https://github.com/andylokandy/tdigests)
    pub fn estimate_rank(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }

        if self.centroids.len() == 1 {
            match self.centroids[0].mean().partial_cmp(&x).unwrap() {
                Ordering::Less => return 1.0,
                Ordering::Equal => return 0.5,
                Ordering::Greater => return 0.0,
            }
        }

        let mut cumulative = 0.0;
        let mut cum_left = 0.0;
        let mut cum_right = 0.0;
        let mut position = 0;

        for (k, centroid) in self.centroids.iter().enumerate() {
            cum_left = cum_right;
            cum_right = (2.0 * cumulative + centroid.weight() - 1.0)
                / 2.0
                / (self.mass() - 1.0);
            cumulative += centroid.weight();

            if centroid.mean() >= x {
                break;
            }

            position = k + 1;
        }

        if position == 0 {
            return 0.0;
        }

        if position >= self.centroids.len() {
            return 1.0;
        }

        let centroid_left = &self.centroids[position - 1];
        let centroid_right = &self.centroids[position];

        let weight_between = cum_right - cum_left;
        let fraction = (x - centroid_left.mean())
            / (centroid_right.mean() - centroid_left.mean());

        cum_left + fraction * weight_between
    }

    pub fn estimate_ranks(
        &self,
        xs: &[f64],
    ) -> Result<Vec<f64>, TryReserveError> {
        let n_centroids = self.centroids.len();

        if n_centroids == 0 {
            return Ok(vec![]);
        }

        if n_centroids == 1 {
            let m = self.centroids[0].mean();
            let ranks = xs
                .iter()
                .map(|&x| {
                    if x.is_nan() {
                        f64::NAN
                    } else {
                        match m.partial_cmp(&x).unwrap() {
                            std::cmp::Ordering::Less => 1.0,
                            std::cmp::Ordering::Equal => 0.5,
                            std::cmp::Ordering::Greater => 0.0,
                        }
                    }
                })
                .collect();
            return Ok(ranks);
        }

        let mut cum_left: Vec<f64> = Vec::new();
        let mut cum_right: Vec<f64> = Vec::new();
        cum_left.try_reserve_exact(n_centroids)?;
        cum_right.try_reserve_exact(n_centroids)?;

        let mut cumulative = 0.0;
        let mut prev_right = 0.0;

        for centroid in &self.centroids {
            let left = prev_right;

            let right = (2.0 * cumulative + centroid.weight() - 1.0)
                / 2.0
                / (self.mass() - 1.0);

            cumulative += centroid.weight();
            prev_right = right;

            cum_left.push(left);
            cum_right.push(right);
        }

        let means: Vec<f64> = self.centroids.iter().map(|c| c.mean()).collect();

        let mut out: Vec<f64> = Vec::new();
        out.try_reserve_exact(xs.len())?;

        for &x in xs {
            if x.is_nan() {
                out.push(f64::NAN);
                continue;
            }
            let idx = means
                .binary_search_by(|m| m.partial_cmp(&x).unwrap())
                .unwrap_or_else(|i| i);

            if idx == 0 {
                out.push(0.0);
                continue;
            }

            if idx >= n_centroids {
                out.push(1.0);
                continue;
            }

            let left_mean = means[idx - 1];
            let right_mean = means[idx];
            let left = cum_left[idx];
            let right = cum_right[idx];
            let weight_between = right - left;

            if right_mean == left_mean {
                out.push(left);
                continue;
            }

            let fraction = (x - left_mean) / (right_mean - left_mean);
            out.push(left + fraction * weight_between);
        }
        Ok(out)
    }

    pub fn estimate_trimmed_mean(&self, q1: f64, q2: f64) -> f64 {
        let lower_weight_threshold = q1 * self.mass();
        let upper_weight_threshold = q2 * self.mass();

        let mut cum_weight = 0.0;
        let mut trimmed_sum = 0.0;
        let mut trimmed_weight = 0.0;

        for centroid in self.centroids().iter() {
            let c_start = cum_weight;
            let c_end = cum_weight + centroid.weight();
            cum_weight = c_end;

            if c_end <= lower_weight_threshold {
                continue;
            }
            if c_start >= upper_weight_threshold {
                break;
            }

            let overlap = (c_end.min(upper_weight_threshold)
                - c_start.max(lower_weight_threshold))
            .max(0.0);
            trimmed_sum += overlap * centroid.mean();
            trimmed_weight += overlap;
        }

        if trimmed_weight == 0.0 {
            return f64::NAN;
        }

        trimmed_sum / trimmed_weight
    }

    pub fn estimate_mad(&self) -> f64 {
        let median = self.estimate_quantile(0.5);

        let mut pairs: Vec<(f64, f64)> = self
            .centroids
            .iter()
            .map(|c| ((c.mean() - median).abs(), c.weight()))
            .collect();

        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        let half = (self.mass() + 1.0) / 2.0;
        let mut cumulative = 0.0;
        let mut prev_cum;
        let mut prev_dev = pairs[0].0;

        for (dev, w) in pairs.into_iter() {
            prev_cum = cumulative;
            cumulative += w;

            if cumulative >= half {
                if cumulative == prev_cum {
                    return dev;
                }
                let frac = (half - prev_cum) / (cumulative - prev_cum);
                return prev_dev * (1.0 - frac) + dev * frac;
            }

            prev_dev = dev;
        }

        self.centroids
            .last()
            .map(|c| (c.mean() - median).abs())
            .unwrap_or(f64::NAN)
    }

    /// Estimates population variance using Var(X) = E[X^2] - (E[X])^2.
    pub fn estimate_var(&self) -> f64 {
        if self.mass() == 0.0 {
            return f64::NAN;
        }
        let m2: f64 = self
            .centroids
            .iter()
            .map(|c| c.mean() * c.mean() * c.weight())
            .sum();
        m2 / self.mass() - self.mean() * self.mean()
    }

    /// Approximate error function (Abramowitz-Stegun 7.1.26).
    fn erf_approx(x: f64) -> f64 {
        let a1: f64 = 0.254829592;
        let a2: f64 = -0.284496736;
        let a3: f64 = 1.421413741;
        let a4: f64 = -1.453152027;
        let a5: f64 = 1.061405429;
        let p: f64 = 0.3275911;

        let x_abs = x.abs();
        let t = 1.0 / (1.0 + p * x_abs);
        let y = 1.0
            - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
                * (-x_abs * x_abs).exp();
        y * x.signum()
    }

    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf_approx(x / 2f64.sqrt()))
    }

    /// Compute a weighted Kolmogorov-Smirnov statistic.
    fn ks_statistic_against_normal(&self) -> f64 {
        let n = self.mass();
        let mu = self.mean();
        let sigma = self.estimate_var().sqrt();

        if sigma == 0.0 || sigma.is_nan() {
            return 1.0;
        }

        let mut cum_before: f64 = 0.0;
        let mut d_max: f64 = 0.0;

        for c in &self.centroids {
            let w = c.weight();
            let mean = c.mean();
            let cum_after = cum_before + w;

            let f_before = cum_before / n;
            let f_after = cum_after / n;

            let z = (mean - mu) / sigma;
            let theo = Self::normal_cdf(z);

            let d1 = (f_after - theo).abs();
            let d2 = (theo - f_before).abs();

            if d1 > d_max {
                d_max = d1;
            }
            if d2 > d_max {
                d_max = d2;
            }

            cum_before = cum_after;
        }
        d_max
    }

    /// Perform a one-sample KS test against a normal distribution.
    pub fn test_cdf_is_normal(&self, alpha: f64) -> bool {
        let d = self.ks_statistic_against_normal();
        let n = self.mass();
        let d_crit = (-0.5 * (alpha / 2.0).ln()).sqrt() / n.sqrt();
        d <= d_crit
    }

    fn maybe_recompute_totals(&mut self, old_count: u128) {
        let old_count_level = old_count / Self::RECOMP_THRESH;
        let new_count_level = self.count / Self::RECOMP_THRESH;
        if new_count_level > old_count_level {
            self.recompute_totals();
        }
    }

    fn recompute_totals(&mut self) {
        let mut mass = 0.0;
        let mut sum = 0.0;
        for c in self.centroids.iter() {
            mass += c.weight();
            sum += c.mean() * c.weight();
        }
        self.mass = OrderedFloat::from(mass);
        self.sum = OrderedFloat::from(sum);
    }
}

#[derive(Debug)]
pub enum BytesError {
    MemError(TryReserveError),
    CorruptData,
    EmptyData,
    WrongArch,
    WrongFormat,
    WrongVersion,
}
