//! T-Digest algorithm in rust
//!
//! ## Installation
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! tdigest = "0.2"
//! ```
//!
//! then you are good to go. If you are using Rust 2015 you have to
//! ``extern crate tdigest`` to your crate root as well.
//!
//! ## Example
//!
//! ```rust
//! use tdigest::TDigest;
//!
//! let t = TDigest::new_with_size(100);
//! let values: Vec<f64> = (1..=1_000_000).map(f64::from).collect();
//!
//! let t = t.merge_sorted(values);
//!
//! let ans = t.estimate_quantile(0.99);
//! let expected: f64 = 990_000.0;
//!
//! let percentage: f64 = (expected - ans).abs() / expected;
//! assert!(percentage < 0.01);
//! ```

use ordered_float::OrderedFloat;
use std::cmp::Ordering;

pub const DEFAULT_MAX_CENTROIDS: usize = 1000;

/// Centroid implementation to the cluster mentioned in the paper.
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

/// T-Digest to be operated on.
#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "use_serde", derive(Serialize, Deserialize))]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    sum: OrderedFloat<f64>,
    count: OrderedFloat<f64>,
    max: OrderedFloat<f64>,
    min: OrderedFloat<f64>,
}

impl TDigest {
    pub fn new_with_size(max_size: usize) -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(std::f64::NAN),
            min: OrderedFloat::from(std::f64::NAN),
        }
    }

    pub fn new(
        centroids: Vec<Centroid>,
        sum: f64,
        count: f64,
        max: f64,
        min: f64,
        max_size: usize,
    ) -> Self {
        if centroids.len() <= max_size {
            TDigest {
                centroids,
                max_size,
                sum: OrderedFloat::from(sum),
                count: OrderedFloat::from(count),
                max: OrderedFloat::from(max),
                min: OrderedFloat::from(min),
            }
        } else {
            let sz = centroids.len();
            let digests: Vec<TDigest> = vec![
                TDigest::new_with_size(max_size),
                TDigest::new(centroids, sum, count, max, min, sz),
            ];
            Self::merge_digests(digests, Some(max_size))
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        let count_: f64 = self.count.into_inner();
        let sum_: f64 = self.sum.into_inner();

        if count_ > 0.0 {
            sum_ / count_
        } else {
            0.0
        }
    }

    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum.into_inner()
    }

    #[inline]
    pub fn count(&self) -> f64 {
        self.count.into_inner()
    }

    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner()
    }

    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
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
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }
}

impl Default for TDigest {
    fn default() -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size: DEFAULT_MAX_CENTROIDS,
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(std::f64::NAN),
            min: OrderedFloat::from(std::f64::NAN),
        }
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

    pub fn merge_unsorted(&self, unsorted_values: Vec<f64>) -> TDigest {
        let mut sorted_values: Vec<OrderedFloat<f64>> = unsorted_values
            .into_iter()
            .map(OrderedFloat::from)
            .collect();
        sorted_values.sort();
        let sorted_values = sorted_values
            .into_iter()
            .map(|f| f.into_inner())
            .collect();

        self.merge_sorted(sorted_values)
    }

    pub fn merge_sorted(&self, sorted_values: Vec<f64>) -> TDigest {
        if sorted_values.is_empty() {
            return self.clone();
        }

        let mut result = TDigest::new_with_size(self.max_size());
        result.count =
            OrderedFloat::from(self.count() + (sorted_values.len() as f64));

        let maybe_min =
            OrderedFloat::from(*sorted_values.first().unwrap());
        let maybe_max =
            OrderedFloat::from(*sorted_values.last().unwrap());

        if self.count() > 0.0 {
            result.min = std::cmp::min(self.min, maybe_min);
            result.max = std::cmp::max(self.max, maybe_max);
        } else {
            result.min = maybe_min;
            result.max = maybe_max;
        }

        let mut compressed: Vec<Centroid> = Vec::with_capacity(self.max_size);

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_count: f64 =
            Self::k_to_q(k_limit, self.max_size as f64)
                * result.count.into_inner();
        k_limit += 1.0;

        let mut iter_centroids =
            self.centroids.iter().peekable();
        let mut iter_sorted_values =
            sorted_values.iter().peekable();

        let mut curr: Centroid = if let Some(c) = iter_centroids.peek()
        {
            let curr = **iter_sorted_values.peek().unwrap();
            if c.mean() < curr {
                iter_centroids.next().unwrap().clone()
            } else {
                Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
            }
        } else {
            Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
        };

        let mut weight_so_far: f64 = curr.weight();

        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        while iter_centroids.peek().is_some()
            || iter_sorted_values.peek().is_some()
        {
            let next: Centroid = if let Some(c) = iter_centroids.peek()
            {
                if iter_sorted_values.peek().is_none()
                    || c.mean() < **iter_sorted_values.peek().unwrap()
                {
                    iter_centroids.next().unwrap().clone()
                } else {
                    Centroid::new(
                        *iter_sorted_values.next().unwrap(),
                        1.0,
                    )
                }
            } else {
                Centroid::new(*iter_sorted_values.next().unwrap(), 1.0)
            };

            let next_sum: f64 = next.mean() * next.weight();
            weight_so_far += next.weight();

            if weight_so_far <= q_limit_times_count {
                sums_to_merge += next_sum;
                weights_to_merge += next.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum.into_inner()
                        + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;

                compressed.push(curr.clone());
                q_limit_times_count =
                    Self::k_to_q(k_limit, self.max_size as f64)
                        * result.count();
                k_limit += 1.0;
                curr = next;
            }
        }

        result.sum = OrderedFloat::from(
            result.sum.into_inner()
                + curr.add(sums_to_merge, weights_to_merge),
        );
        compressed.push(curr);
        compressed.shrink_to_fit();
        compressed.sort();

        result.centroids = compressed;
        result
    }

    fn external_merge(
        centroids: &mut Vec<Centroid>,
        first: usize,
        middle: usize,
        last: usize,
    ) {
        let mut result: Vec<Centroid> = Vec::with_capacity(centroids.len());

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
    }

    // Merge multiple T-Digests
    pub fn merge_digests(
        digests: Vec<TDigest>,
        max_size: Option<usize>,
    ) -> TDigest {
        let max_size = if let Some(max) = max_size {
            max
        } else {
            digests
                .iter()
                .map(|digest| digest.max_size)
                .max()
                .unwrap_or(DEFAULT_MAX_CENTROIDS)
        };

        let n_centroids: usize =
            digests.iter().map(|d| d.centroids.len()).sum();
        if n_centroids == 0 {
            return TDigest::new_with_size(max_size);
        }

        let mut centroids: Vec<Centroid> = Vec::with_capacity(n_centroids);
        let mut starts: Vec<usize> = Vec::with_capacity(digests.len());

        let mut count: f64 = 0.0;
        let mut min = OrderedFloat::from(std::f64::INFINITY);
        let mut max = OrderedFloat::from(std::f64::NEG_INFINITY);

        let mut start: usize = 0;
        for digest in digests.into_iter() {
            starts.push(start);

            let curr_count: f64 = digest.count();
            if curr_count > 0.0 {
                min = std::cmp::min(min, digest.min);
                max = std::cmp::max(max, digest.max);
                count += curr_count;
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
                    let last =
                        if i + 2 * digests_per_block < starts.len() {
                            starts[i + 2 * digests_per_block]
                        } else {
                            centroids.len()
                        };

                    debug_assert!(first <= middle && middle <= last);
                    Self::external_merge(&mut centroids, first, middle, last);
                }
            }

            digests_per_block *= 2;
        }

        let mut result = TDigest::new_with_size(max_size);
        let mut compressed: Vec<Centroid> = Vec::with_capacity(max_size);

        let mut k_limit: f64 = 1.0;
        let mut q_limit_times_count: f64 =
            Self::k_to_q(k_limit, max_size as f64) * count;

        let mut iter_centroids = centroids.iter_mut();
        let mut curr = iter_centroids.next().unwrap();
        let mut weight_so_far: f64 = curr.weight();
        let mut sums_to_merge: f64 = 0.0;
        let mut weights_to_merge: f64 = 0.0;

        for centroid in iter_centroids {
            weight_so_far += centroid.weight();

            if weight_so_far <= q_limit_times_count {
                sums_to_merge += centroid.mean() * centroid.weight();
                weights_to_merge += centroid.weight();
            } else {
                result.sum = OrderedFloat::from(
                    result.sum.into_inner()
                        + curr.add(sums_to_merge, weights_to_merge),
                );
                sums_to_merge = 0.0;
                weights_to_merge = 0.0;
                compressed.push(curr.clone());
                q_limit_times_count =
                    Self::k_to_q(k_limit, max_size as f64) * count;
                k_limit += 1.0;
                curr = centroid;
            }
        }

        result.sum = OrderedFloat::from(
            result.sum.into_inner()
                + curr.add(sums_to_merge, weights_to_merge),
        );
        compressed.push(curr.clone());
        compressed.shrink_to_fit();
        compressed.sort();

        result.count = OrderedFloat::from(count);
        result.min = min;
        result.max = max;
        result.centroids = compressed;
        result
    }

    /// Function by Andy Lok (https://github.com/andylokandy/tdigests)
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        let q = q.clamp(0.0, 1.0);

        if self.centroids.len() == 1 {
            return self.centroids[0].mean.into_inner();
        }

        let mut cumulative = 0.0;
        let mut cum_left = 0.0;
        let mut cum_right = 0.0;
        let mut position = 0;

        for (k, centroid) in self.centroids.iter().enumerate() {
            cum_left = cum_right;
            cum_right =
                (2.0 * cumulative + centroid.weight.into_inner() - 1.0)
                    / 2.0
                    / (self.count() - 1.0);
            cumulative += centroid.weight.into_inner();

            if cum_right >= q {
                break;
            }

            position = k + 1;
        }

        if position == 0 {
            return self.centroids[0].mean.into_inner();
        }

        if position >= self.centroids.len() {
            return self.centroids[self.centroids.len() - 1].mean.into_inner();
        }

        let centroid_left = &self.centroids[position - 1];
        let centroid_right = &self.centroids[position];

        let weight_between = cum_right - cum_left;
        let fraction = (q - cum_left) / weight_between;

        centroid_left.mean.into_inner() * (1.0 - fraction)
            + centroid_right.mean.into_inner() * fraction
    }

    /// Function by Andy Lok (https://github.com/andylokandy/tdigests)
    pub fn estimate_rank(&self, x: f64) -> f64 {
        if self.centroids.len() == 1 {
            match self.centroids[0]
                .mean
                .into_inner()
                .partial_cmp(&x)
                .unwrap()
            {
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
            cum_right =
                (2.0 * cumulative + centroid.weight.into_inner() - 1.0)
                    / 2.0
                    / (self.count() - 1.0);
            cumulative += centroid.weight.into_inner();

            if centroid.mean.into_inner() >= x {
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
        let fraction = (x - centroid_left.mean.into_inner())
            / (centroid_right.mean.into_inner()
                - centroid_left.mean.into_inner());

        cum_left + fraction * weight_between
    }
}
