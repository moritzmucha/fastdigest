mod tdigest;

use parking_lot::{Mutex, MutexGuard};
use pyo3::exceptions::{PyKeyError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::TryReserveError;
use std::mem;
use tdigest::{
    BytesError, Centroid, TDigest, TD_SIZE_DEFAULT, TD_SIZE_PLATFORM_MAX,
};

const CACHE_SIZE: usize = 256;

#[derive(Clone)]
struct TDigestState {
    digest: TDigest,
    x_cache: [f64; CACHE_SIZE],
    w_cache: [f64; CACHE_SIZE],
    w_cache_set: bool,
    i: usize,
}

impl Default for TDigestState {
    fn default() -> Self {
        let digest: TDigest = TDigest::new_with_size(TD_SIZE_DEFAULT)
            .expect("default max size should be allocatable");
        Self {
            digest,
            x_cache: [0.0; CACHE_SIZE],
            w_cache: [1.0; CACHE_SIZE],
            w_cache_set: false,
            i: 0,
        }
    }
}

#[pyclass(name = "TDigest", module = "fastdigest")]
pub struct PyTDigest {
    state: Mutex<TDigestState>,
}

impl Clone for PyTDigest {
    fn clone(&self) -> Self {
        let state = self.state.lock().clone();
        Self {
            state: Mutex::new(state),
        }
    }
}

#[pymethods]
impl PyTDigest {
    /// Constructs a new empty TDigest instance.
    #[new]
    #[pyo3(signature = (max_centroids=TD_SIZE_DEFAULT as i64))]
    pub fn new(max_centroids: i64) -> PyResult<Self> {
        let max_cent_valid = validate_max_centroids(max_centroids)?;
        let digest =
            TDigest::new_with_size(max_cent_valid).map_err(malloc_error)?;
        Ok(Self {
            state: Mutex::new(TDigestState {
                digest,
                ..TDigestState::default()
            }),
        })
    }

    /// Constructs a new TDigest from a sequence of float values.
    #[staticmethod]
    #[pyo3(signature = (x, w=None, max_centroids=TD_SIZE_DEFAULT as i64))]
    pub fn from_values(
        x: Vec<f64>,
        w: Option<Bound<'_, PyAny>>,
        max_centroids: i64,
    ) -> PyResult<Self> {
        let max_cent_valid = validate_max_centroids(max_centroids)?;
        let digest =
            TDigest::new_with_size(max_cent_valid).map_err(malloc_error)?;
        if x.is_empty() {
            Ok(Self {
                state: Mutex::new(TDigestState {
                    digest,
                    ..TDigestState::default()
                }),
            })
        } else {
            validate_values(&x)?;
            let w_vec = validate_weights(w, x.len())?;
            let digest = match w_vec {
                Some(weights) => digest
                    .merge_unsorted_weighted(x, weights)
                    .map_err(malloc_error)?,
                None => digest.merge_unsorted(x).map_err(malloc_error)?,
            };
            Ok(Self {
                state: Mutex::new(TDigestState {
                    digest,
                    ..TDigestState::default()
                }),
            })
        }
    }

    /// Reconstructs a TDigest from its binary representation.
    #[staticmethod]
    pub fn from_bytes(data: &[u8]) -> PyResult<Self> {
        match TDigest::from_bytes(data) {
            Ok(digest) => Ok(Self {
                state: Mutex::new(TDigestState {
                    digest,
                    ..TDigestState::default()
                }),
            }),
            Err(BytesError::MemError(e)) => Err(malloc_error(e)),
            Err(BytesError::CorruptData) => {
                Err(PyValueError::new_err("Data is corrupt."))
            }
            Err(BytesError::EmptyData) => {
                Err(PyValueError::new_err("Data is empty."))
            }
            Err(BytesError::WrongArch) => Err(PyValueError::new_err(
                "Data requires 64-bit architecture to load into TDigest.",
            )),
            Err(BytesError::WrongFormat) => Err(PyValueError::new_err(
                "Data is not in fastDigest binary format.",
            )),
            Err(BytesError::WrongVersion) => {
                Err(PyValueError::new_err(format!(
                    "Data format version is incompatible with fastDigest v{}",
                    env!("CARGO_PKG_VERSION")
                )))
            }
        }
    }

    /// Reconstructs a TDigest from a dict.
    #[staticmethod]
    pub fn from_dict(tdigest_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let centroids_obj =
            tdigest_dict.get_item("centroids")?.ok_or_else(|| {
                PyKeyError::new_err("Key 'centroids' not found in dict.")
            })?;
        let centroids_list = centroids_obj.cast::<PyList>()?;
        let mut centroids: Vec<Centroid> = Vec::new();
        centroids
            .try_reserve_exact(centroids_list.len())
            .map_err(malloc_error)?;
        let mut sum = 0.0;
        let mut mass = 0.0;
        let mut min = f64::NAN;
        let mut max = f64::NAN;

        for item in centroids_list.iter() {
            let d = item.cast::<PyDict>()?;
            let mean: f64 = d
                .get_item("m")?
                .ok_or_else(|| {
                    PyKeyError::new_err("Centroid missing 'm' key.")
                })?
                .extract()?;
            let weight: f64 = d
                .get_item("c")?
                .ok_or_else(|| {
                    PyKeyError::new_err("Centroid missing 'c' key.")
                })?
                .extract()?;
            centroids.push(Centroid::new(mean, weight));
            sum += mean * weight;
            mass += weight;
            min = min.min(mean);
            max = max.max(mean);
        }

        let max_centroids: usize =
            match tdigest_dict.get_item("max_centroids")? {
                Some(obj) => validate_max_centroids(obj.extract::<i64>()?)?,
                _ => TD_SIZE_DEFAULT,
            };
        let mass: f64 = match tdigest_dict.get_item("mass")? {
            Some(obj) => obj.extract()?,
            _ => mass,
        };
        let sum: f64 = match tdigest_dict.get_item("sum")? {
            Some(obj) => obj.extract()?,
            _ => sum,
        };
        let min: f64 = match tdigest_dict.get_item("min")? {
            Some(obj) => obj.extract()?,
            _ => min,
        };
        let max: f64 = match tdigest_dict.get_item("max")? {
            Some(obj) => obj.extract()?,
            _ => max,
        };
        let n_values: u128 = match tdigest_dict.get_item("n_values")? {
            Some(obj) => obj.extract()?,
            _ => mass.round() as u128,
        };

        let digest = if !centroids.is_empty() {
            TDigest::new(
                centroids,
                max_centroids,
                mass,
                sum,
                min,
                max,
                n_values,
            )
            .map_err(malloc_error)?
        } else {
            TDigest::new_with_size(max_centroids).map_err(malloc_error)?
        };

        Ok(Self {
            state: Mutex::new(TDigestState {
                digest,
                ..TDigestState::default()
            }),
        })
    }

    /// Getter property: returns the max_centroids parameter.
    #[getter(max_centroids)]
    pub fn get_max_centroids(&self) -> PyResult<usize> {
        Ok(lock_state(self)?.digest.max_size())
    }

    /// Setter property: sets the max_centroids parameter.
    #[setter(max_centroids)]
    pub fn set_max_centroids(&self, max_centroids: i64) -> PyResult<()> {
        let max_cent_valid = validate_max_centroids(max_centroids)?;
        lock_state(self)?.digest.set_max_size(max_cent_valid);
        Ok(())
    }

    /// Getter property: returns the total weight of data points ingested.
    #[getter(mass)]
    pub fn get_mass(&self) -> PyResult<f64> {
        let state = lock_state(self)?;
        let w_cache_sum = if state.w_cache_set {
            Vec::from(&state.w_cache[0..state.i]).iter().sum()
        } else {
            state.i as f64
        };
        Ok(state.digest.mass() + w_cache_sum)
    }

    /// Getter property: returns the total number of data points ingested.
    #[getter(n_values)]
    pub fn get_n_values(&self) -> PyResult<u128> {
        let state = lock_state(self)?;
        Ok(state.digest.count() + state.i as u128)
    }

    /// Getter property: returns the number of centroids.
    #[getter(n_centroids)]
    pub fn get_n_centroids(&self) -> PyResult<usize> {
        let state = lock_and_flush(self)?;
        Ok(state.digest.centroids().len())
    }

    /// Getter property: returns True if the digest is empty.
    #[getter(is_empty)]
    pub fn get_is_empty(&self) -> PyResult<bool> {
        let state = lock_state(self)?;
        Ok(state.digest.is_empty() && (state.i == 0))
    }

    /// Getter property: returns the centroids as a list of tuples.
    #[getter(centroids)]
    pub fn get_centroids<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = lock_and_flush(self)?;
        let centroid_list = PyList::empty(py);
        for centroid in state.digest.centroids() {
            let t = PyTuple::new(py, [centroid.mean(), centroid.weight()])?;
            centroid_list.append(t)?;
        }
        Ok(centroid_list)
    }

    /// Merges this digest with another, returning a new TDigest.
    pub fn merge(&self, other: &Self) -> PyResult<Self> {
        let (first, second) = order_by_address(self, other);
        let digest1 = lock_and_flush(first)?.digest.clone();
        let digest2 = lock_and_flush(second)?.digest.clone();
        let digests: Vec<TDigest> = vec![digest1, digest2];
        let merged =
            TDigest::merge_digests(digests, None).map_err(malloc_error)?;
        Ok(Self {
            state: Mutex::new(TDigestState {
                digest: merged,
                ..TDigestState::default()
            }),
        })
    }

    /// Merges this digest with another, modifying the current instance.
    pub fn merge_inplace(&self, other: &Self) -> PyResult<()> {
        let self_addr = self as *const _ as usize;
        let other_addr = other as *const _ as usize;

        if self_addr == other_addr {
            // same object -> clone digest from already-locked state
            let mut state = lock_and_flush(self)?;
            let max_size = state.digest.max_size();
            let lhs = mem::take(&mut state.digest);
            let other_digest = lhs.clone();
            let digests = vec![lhs, other_digest];
            state.digest = TDigest::merge_digests(digests, Some(max_size))
                .map_err(malloc_error)?;
            Ok(())
        } else if self_addr < other_addr {
            // lock self first, then other
            let mut state = lock_and_flush(self)?;
            let other_digest = lock_and_flush(other)?.digest.clone();
            let max_size = state.digest.max_size();
            let lhs = mem::take(&mut state.digest);
            let digests = vec![lhs, other_digest];
            state.digest = TDigest::merge_digests(digests, Some(max_size))
                .map_err(malloc_error)?;
            Ok(())
        } else {
            // lock other first, then self
            let other_digest = lock_and_flush(other)?.digest.clone();
            let mut state = lock_and_flush(self)?;
            let max_size = state.digest.max_size();
            let lhs = mem::take(&mut state.digest);
            let digests = vec![lhs, other_digest];
            state.digest = TDigest::merge_digests(digests, Some(max_size))
                .map_err(malloc_error)?;
            Ok(())
        }
    }

    /// Updates the digest (in-place) with a sequence of float values.
    #[pyo3(signature = (x, w=None))]
    pub fn batch_update(
        &self,
        x: Vec<f64>,
        w: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if x.is_empty() {
            return Ok(());
        }

        validate_values(&x)?;
        let w_vec = validate_weights(w, x.len())?;
        let mut state = lock_and_flush(self)?;
        state.digest = match w_vec {
            Some(weights) => state
                .digest
                .merge_unsorted_weighted(x, weights)
                .map_err(malloc_error)?,
            None => state.digest.merge_unsorted(x).map_err(malloc_error)?,
        };
        Ok(())
    }

    /// Updates the digest (in-place) with a single float value.
    #[inline]
    #[pyo3(signature = (x, w=None))]
    pub fn update(&self, x: f64, w: Option<f64>) -> PyResult<()> {
        validate_value(x)?;
        let weight = validate_weight(w.unwrap_or(1.0))?;
        let mut state = lock_state(self)?;
        record_observation(&mut state, x, weight)?;
        Ok(())
    }

    /// Estimates the quantile for a given cumulative probability `q`.
    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("q must be between 0 and 1."));
        }
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_quantile(q))
    }

    /// Estimates the quantiles for given cumulative probabilities `q`.
    pub fn quantile_vec(&self, q: Vec<f64>) -> PyResult<Vec<f64>> {
        if q.iter().any(|q_i| !(0.0..=1.0).contains(q_i)) {
            return Err(PyValueError::new_err(
                "All q values must be between 0 and 1.",
            ));
        }
        let state = lock_flush_check(self)?;
        let d = &state.digest;
        let x = match q.len() {
            0 => vec![],
            1 | 2 => q.iter().map(|&q_i| d.estimate_quantile(q_i)).collect(),
            _ => d.estimate_quantiles(&q).map_err(malloc_error)?,
        };
        Ok(x)
    }

    /// Estimates the percentile for a given cumulative probability `p` (%).
    pub fn percentile(&self, p: f64) -> PyResult<f64> {
        if !(0.0..=100.0).contains(&p) {
            return Err(PyValueError::new_err("p must be between 0 and 100."));
        }
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_quantile(0.01 * p))
    }

    /// Estimates the median.
    pub fn median(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_quantile(0.5))
    }

    /// Estimates the inter-quartile range.
    pub fn iqr(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        let d = &state.digest;
        Ok(d.estimate_quantile(0.75) - d.estimate_quantile(0.25))
    }

    /// Estimates the rank (cumulative probability) of a given value `x`.
    pub fn cdf(&self, x: f64) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_rank(x))
    }

    /// Estimates the ranks (cumulative probabilities) of given values `x`.
    pub fn cdf_vec(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        let state = lock_flush_check(self)?;
        let d = &state.digest;
        let q = match x.len() {
            0 => vec![],
            1 | 2 => x.iter().map(|&x_i| d.estimate_rank(x_i)).collect(),
            _ => d.estimate_ranks(&x).map_err(malloc_error)?,
        };
        Ok(q)
    }

    /// Estimates the empirical probability of a value being in
    /// the interval \[`x1`, `x2`\].
    pub fn probability(&self, x1: f64, x2: f64) -> PyResult<f64> {
        if x1 > x2 {
            return Err(PyValueError::new_err(
                "x1 must be less than or equal to x2.",
            ));
        }
        let state = lock_flush_check(self)?;
        let d = &state.digest;
        Ok(d.estimate_rank(x2) - d.estimate_rank(x1))
    }

    /// Returns the sum of the data.
    pub fn sum(&self) -> PyResult<f64> {
        let state = lock_and_flush(self)?;
        Ok(state.digest.sum())
    }

    /// Returns the mean of the data.
    pub fn mean(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.mean())
    }

    /// Returns the trimmed mean of the data between the q1 and q2 quantiles.
    pub fn trimmed_mean(&self, q1: f64, q2: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&q1) || !(0.0..=1.0).contains(&q2) || q1 >= q2
        {
            return Err(PyValueError::new_err(
                "q1 must be >= 0, q2 must be <= 1, and q1 < q2.",
            ));
        }
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_trimmed_mean(q1, q2))
    }

    /// Returns the lowest ingested value.
    pub fn min(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.min())
    }

    /// Returns the highest ingested value.
    pub fn max(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.max())
    }

    /// Estimates the median absolute deviation.
    pub fn mad(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_mad())
    }

    /// Estimates the variance.
    pub fn var(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_var())
    }

    /// Estimates the standard deviation.
    pub fn std(&self) -> PyResult<f64> {
        let state = lock_flush_check(self)?;
        Ok(state.digest.estimate_var().sqrt())
    }

    /// Performs a KS test to determine normality.
    #[pyo3(signature = (alpha=0.05))]
    pub fn is_normal(&self, alpha: f64) -> PyResult<bool> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(PyValueError::new_err(
                "alpha must be strictly greater than 0 and less than 1.",
            ));
        }
        let state = lock_flush_check(self)?;
        Ok(state.digest.test_cdf_is_normal(alpha))
    }

    /// Returns a binary representation of the digest.
    pub fn to_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let state = lock_and_flush(self)?;
        let bytes = state.digest.to_bytes().map_err(malloc_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Returns a dict representation of the digest.
    pub fn to_dict<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let state = lock_and_flush(self)?;
        let dict = PyDict::new(py);

        dict.set_item("max_centroids", state.digest.max_size())?;
        dict.set_item("mass", state.digest.mass())?;
        dict.set_item("sum", state.digest.sum())?;
        dict.set_item("min", state.digest.min())?;
        dict.set_item("max", state.digest.max())?;
        dict.set_item("n_values", state.digest.count())?;

        let centroid_list = PyList::empty(py);
        for centroid in state.digest.centroids() {
            let centroid_dict = PyDict::new(py);
            centroid_dict.set_item("m", centroid.mean())?;
            centroid_dict.set_item("c", centroid.weight())?;
            centroid_list.append(centroid_dict)?;
        }
        dict.set_item("centroids", centroid_list)?;
        Ok(dict)
    }

    /// Returns true if two digests are equal. Caches are flushed
    /// to ensure accurate results across disparate states.
    pub fn equals(&self, other: &Self) -> PyResult<bool> {
        if std::ptr::eq(self, other) {
            return Ok(true);
        }

        fn summary_equal(d1: &TDigest, d2: &TDigest) -> bool {
            (d1.max_size() == d2.max_size())
                && (d1.mass() == d2.mass())
                && (d1.sum() == d2.sum())
                && ((d1.min().is_nan() && d2.min().is_nan())
                    || (d1.min() == d2.min()))
                && ((d1.max().is_nan() && d2.max().is_nan())
                    || (d1.max() == d2.max()))
                && (d1.count() == d2.count())
        }

        fn centroids_equal(c1: &Centroid, c2: &Centroid) -> bool {
            (c1.mean() == c2.mean()) && (c1.weight() == c2.weight())
        }

        let (first, second) = order_by_address(self, other);
        let digest1 = &lock_and_flush(first)?.digest;
        let digest2 = &lock_and_flush(second)?.digest;
        let cents1 = digest1.centroids();
        let cents2 = digest2.centroids();

        if !summary_equal(digest1, digest2) {
            return Ok(false);
        }
        if cents1.len() != cents2.len() {
            return Ok(false);
        }
        for (c1, c2) in cents1.iter().zip(cents2.iter()) {
            if !centroids_equal(c1, c2) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// TDigest.copy() returns a copy of the instance.
    pub fn copy(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Magic method: copy(digest) returns a copy of the instance.
    pub fn __copy__(&self) -> PyResult<Self> {
        self.copy()
    }

    /// Magic method: deepcopy(digest) returns a copy of the instance.
    pub fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.copy()
    }

    /// Returns a tuple (callable, args) so that pickle can reconstruct
    /// the object via TDigest.from_bytes(state)
    pub fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let bytes = self.to_bytes(py)?;
        let cls = py.get_type::<PyTDigest>();
        let from_bytes = cls.getattr("from_bytes")?;
        let args = PyTuple::new(py, &[bytes])?;
        let recon = PyTuple::new(py, &[from_bytes, args.into_any()])?;
        Ok(recon)
    }

    /// Magic method: bool(TDigest) returns the negation of is_empty().
    pub fn __bool__(&self) -> PyResult<bool> {
        self.get_is_empty().map(|empty| !empty)
    }

    /// Magic method: len(TDigest) returns the number of centroids.
    pub fn __len__(&self) -> PyResult<usize> {
        self.get_n_centroids()
    }

    // Magic method: returns an iterator over the list of centroids.
    pub fn __iter__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let centroid_list = self.get_centroids(py)?;
        centroid_list.call_method0("__iter__")
    }

    /// Magic method: repr/str(TDigest) returns a string representation.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TDigest(max_centroids={})",
            lock_state(self)?.digest.max_size()
        ))
    }

    /// Magic method: enables equality checking (==).
    pub fn __eq__(&self, other: &Self) -> PyResult<bool> {
        self.equals(other)
    }

    /// Magic method: enables inequality checking (!=).
    pub fn __ne__(&self, other: &Self) -> PyResult<bool> {
        self.equals(other).map(|eq| !eq)
    }

    /// Magic method: dig1 + dig2 returns dig1.merge(dig2).
    pub fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.merge(other)
    }

    /// Magic method: dig1 += dig2 calls dig1.merge_inplace(dig2).
    pub fn __iadd__(&self, other: &Self) -> PyResult<()> {
        self.merge_inplace(other)
    }
}

/// Top-level function for more efficient merging of many TDigest instances.
#[pyfunction]
#[pyo3(signature = (digests, max_centroids=None))]
pub fn merge_all(
    digests: &Bound<'_, PyAny>,
    max_centroids: Option<i64>,
) -> PyResult<PyTDigest> {
    let digests: Vec<TDigest> = digests
        .try_iter()?
        .map(|item| {
            let py_tdigest =
                item.and_then(|x| x.extract::<PyTDigest>()).map_err(|_| {
                    PyTypeError::new_err("Provide an iterable of TDigests.")
                })?;
            let state = lock_and_flush(&py_tdigest)?;
            Ok(state.digest.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let max_cent_valid: Option<usize> = match max_centroids {
        Some(v) => Some(validate_max_centroids(v)?),
        None => None,
    };

    let merged = TDigest::merge_digests(digests, max_cent_valid)
        .map_err(malloc_error)?;
    Ok(PyTDigest {
        state: Mutex::new(TDigestState {
            digest: merged,
            ..TDigestState::default()
        }),
    })
}

/// Online TDigest algorithm by kvc0 (https://github.com/MnO2/t-digest/pull/2)
#[inline]
fn record_observation(
    state: &mut TDigestState,
    observation: f64,
    weight: f64,
) -> PyResult<()> {
    state.x_cache[state.i] = observation;
    if weight != 1.0 {
        state.w_cache[state.i] = weight;
        state.w_cache_set = true;
    }
    state.i += 1;
    if state.i == CACHE_SIZE {
        flush_cache(state)?;
    }
    Ok(())
}

/// Online TDigest algorithm by kvc0 (https://github.com/MnO2/t-digest/pull/2)
#[inline]
fn flush_cache(state: &mut TDigestState) -> PyResult<()> {
    if state.i < 1 {
        return Ok(());
    }
    let x = Vec::from(&state.x_cache[0..state.i]);
    if state.w_cache_set {
        let w = Vec::from(&state.w_cache[0..state.i]);
        state.digest = state
            .digest
            .merge_unsorted_weighted(x, w)
            .map_err(malloc_error)?;
        state.w_cache = [1.0; CACHE_SIZE];
        state.w_cache_set = false;
    } else {
        state.digest = state.digest.merge_unsorted(x).map_err(malloc_error)?;
    }
    state.i = 0;
    Ok(())
}

/// Helper function to raise ValueError on empty digests
#[inline]
fn check_nonempty(state: &TDigestState) -> PyResult<()> {
    if state.digest.is_empty() {
        Err(PyValueError::new_err("TDigest is empty."))
    } else {
        Ok(())
    }
}

/// Helper function for mutex acquisition
#[inline]
fn lock_state(pytd: &PyTDigest) -> PyResult<MutexGuard<'_, TDigestState>> {
    Ok(pytd.state.lock())
}

/// Helper function to `lock_state` + `flush_cache`
#[inline]
fn lock_and_flush(pytd: &PyTDigest) -> PyResult<MutexGuard<'_, TDigestState>> {
    let mut state = lock_state(pytd)?;
    flush_cache(&mut state)?;
    Ok(state)
}

/// Helper function to `lock_state` + `flush_cache` + `check_nonempty`
#[inline]
fn lock_flush_check(
    pytd: &PyTDigest,
) -> PyResult<MutexGuard<'_, TDigestState>> {
    let state = lock_and_flush(pytd)?;
    check_nonempty(&state)?;
    Ok(state)
}

/// Helper function to determine lock order based on pointer addresses
#[inline]
fn order_by_address<'a>(
    first: &'a PyTDigest,
    second: &'a PyTDigest,
) -> (&'a PyTDigest, &'a PyTDigest) {
    if (first as *const _) < (second as *const _) {
        (first, second)
    } else {
        (second, first)
    }
}

/// Helper function to safely convert max_centroids to usize
fn validate_max_centroids(max_centroids: i64) -> PyResult<usize> {
    let max_centroids_usize = usize::try_from(max_centroids).map_err(|_| {
        PyValueError::new_err("max_centroids must be a non-negative integer.")
    })?;
    if max_centroids_usize > TD_SIZE_PLATFORM_MAX {
        return Err(PyValueError::new_err(
            "max_centroids exceeds the platform limit.",
        ));
    }
    Ok(max_centroids_usize)
}

#[inline]
fn validate_value(value: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("Values must be finite."));
    }
    Ok(value)
}

#[inline]
fn validate_values(values: &[f64]) -> PyResult<()> {
    for &x in values {
        validate_value(x)?;
    }
    Ok(())
}

#[inline]
fn validate_weight(weight: f64) -> PyResult<f64> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(PyValueError::new_err(
            "Weights must be finite and greater than 0.",
        ));
    }
    Ok(weight)
}

/// Helper function to validate `w`. If scalar, creates Vec of length `x_len`.
#[inline]
fn validate_weights(
    w: Option<Bound<'_, PyAny>>,
    x_len: usize,
) -> PyResult<Option<Vec<f64>>> {
    match w {
        None => Ok(None),
        Some(obj) => {
            if let Ok(single_weight) = obj.extract::<f64>() {
                let w = validate_weight(single_weight)?;
                return Ok(Some(vec![w; x_len]));
            }

            let w_vec: Vec<f64> = obj.extract::<Vec<f64>>().map_err(|_| {
                PyTypeError::new_err(
                    "w (weight) must be a number or sequence of numbers.",
                )
            })?;
            if w_vec.len() != x_len {
                return Err(PyValueError::new_err(
                    "w (weight) sequence must have the same length as x.",
                ));
            }
            for &w in &w_vec {
                validate_weight(w)?;
            }
            Ok(Some(w_vec))
        }
    }
}

/// Helper function to raise memory allocation errors
#[cold]
fn malloc_error(_err: TryReserveError) -> PyErr {
    PyMemoryError::new_err("Failed to allocate sufficient memory for TDigest.")
}

/// Python module definition
#[pymodule(gil_used = false)]
fn fastdigest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(merge_all, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
