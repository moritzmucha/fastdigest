mod tdigest;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use tdigest::{Centroid, TDigest, DEFAULT_MAX_CENTROIDS};

const CACHE_SIZE: u8 = 255;

#[pyclass(name = "TDigest", module = "fastdigest")]
#[derive(Clone)]
pub struct PyTDigest {
    digest: TDigest,
    cache: [f64; CACHE_SIZE as usize],
    i: u8,
}

impl Default for PyTDigest {
    fn default() -> Self {
        Self {
            digest: TDigest::new_with_size(DEFAULT_MAX_CENTROIDS),
            cache: [0.0; CACHE_SIZE as usize],
            i: 0,
        }
    }
}

#[pymethods]
impl PyTDigest {
    /// Constructs a new empty TDigest instance.
    #[new]
    #[pyo3(signature = (max_centroids=DEFAULT_MAX_CENTROIDS))]
    pub fn new(max_centroids: usize) -> PyResult<Self> {
        Ok(Self {
            digest: TDigest::new_with_size(max_centroids),
            ..Default::default()
        })
    }

    /// Constructs a new TDigest from a sequence of float values.
    #[staticmethod]
    #[pyo3(signature = (values, max_centroids=DEFAULT_MAX_CENTROIDS))]
    pub fn from_values(
        values: Vec<f64>,
        max_centroids: usize,
    ) -> PyResult<Self> {
        let digest = TDigest::new_with_size(max_centroids);
        if values.is_empty() {
            Ok(Self {
                digest,
                ..Default::default()
            })
        } else {
            let digest = digest.merge_unsorted(values);
            Ok(Self {
                digest,
                ..Default::default()
            })
        }
    }

    /// Getter property: returns the max_centroids parameter.
    #[getter(max_centroids)]
    pub fn get_max_centroids(&self) -> PyResult<usize> {
        Ok(self.digest.max_size())
    }

    /// Setter property: sets the max_centroids parameter.
    #[setter(max_centroids)]
    pub fn set_max_centroids(&mut self, max_centroids: usize) {
        self.digest.set_max_size(max_centroids);
    }

    /// Getter property: returns the total number of data points ingested.
    #[getter(n_values)]
    pub fn get_n_values(&self) -> PyResult<u64> {
        Ok(self.digest.count().round() as u64 + self.i as u64)
    }

    /// Getter property: returns the number of centroids.
    #[getter(n_centroids)]
    pub fn get_n_centroids(&mut self) -> PyResult<usize> {
        flush_cache(self);
        Ok(self.digest.centroids().len())
    }

    /// Getter property: returns True if the digest is empty.
    #[getter(is_empty)]
    pub fn get_is_empty(&self) -> PyResult<bool> {
        Ok(self.digest.is_empty() && (self.i == 0))
    }

    /// Getter property: returns the centroids as a list of tuples.
    #[getter(centroids)]
    pub fn get_centroids(&mut self, py: Python) -> PyResult<PyObject> {
        flush_cache(self);

        let centroid_list = PyList::empty(py);
        for centroid in self.digest.centroids() {
            let tuple = PyTuple::new(
                py,
                &[centroid.mean.into_inner(), centroid.weight.into_inner()]
            )?;
            centroid_list.append(tuple)?;
        }
        Ok(centroid_list.into())
    }

    /// Merges this digest with another, returning a new TDigest.
    pub fn merge(&mut self, other: &mut Self) -> PyResult<Self> {
        flush_cache(self);
        flush_cache(other);

        let digests: Vec<TDigest> =
            vec![self.digest.clone(), other.digest.clone()];
        let merged = TDigest::merge_digests(digests, None);
        Ok(Self {
            digest: merged,
            ..Default::default()
        })
    }

    /// Merges this digest with another, modifying the current instance.
    pub fn merge_inplace(&mut self, other: &mut Self) {
        flush_cache(self);
        flush_cache(other);

        let digests: Vec<TDigest> =
            vec![self.digest.clone(), other.digest.clone()];
        self.digest =
            TDigest::merge_digests(digests, Some(self.digest.max_size()));
    }

    /// Updates the digest (in-place) with a sequence of float values.
    pub fn batch_update(&mut self, values: Vec<f64>) {
        flush_cache(self);

        if values.is_empty() {
            return;
        }
        self.digest = self.digest.merge_unsorted(values);
    }

    /// Updates the digest (in-place) with a single float value.
    #[inline]
    pub fn update(&mut self, value: f64) {
        record_observation(self, value);
    }

    /// Estimates the quantile for a given cumulative probability `q`.
    pub fn quantile(&mut self, q: f64) -> PyResult<f64> {
        flush_cache(self);

        if q < 0.0 || q > 1.0 {
            return Err(PyValueError::new_err("q must be between 0 and 1."));
        }
        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.estimate_quantile(q))
    }

    /// Estimates the percentile for a given cumulative probability `p` (%).
    pub fn percentile(&mut self, p: f64) -> PyResult<f64> {
        flush_cache(self);

        if p < 0.0 || p > 100.0 {
            return Err(PyValueError::new_err("p must be between 0 and 100."));
        }
        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.estimate_quantile(0.01 * p))
    }

    /// Estimates the rank (cumulative probability) of a given value `x`.
    pub fn cdf(&mut self, x: f64) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.estimate_rank(x))
    }

    /// Returns the trimmed mean of the data between the q1 and q2 quantiles.
    pub fn trimmed_mean(&mut self, q1: f64, q2: f64) -> PyResult<f64> {
        flush_cache(self);

        if q1 < 0.0 || q2 > 1.0 || q1 >= q2 {
            return Err(PyValueError::new_err(
                "q1 must be >= 0, q2 must be <= 1, and q1 < q2.",
            ));
        }
        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }

        let centroids = self.digest.centroids();
        let total_weight: f64 =
            centroids.iter().map(|c| c.weight.into_inner()).sum();
        if total_weight == 0.0 {
            return Err(PyValueError::new_err("Total weight is zero."));
        }
        let lower_weight_threshold = q1 * total_weight;
        let upper_weight_threshold = q2 * total_weight;

        let mut cum_weight = 0.0;
        let mut trimmed_sum = 0.0;
        let mut trimmed_weight = 0.0;
        for centroid in centroids {
            let c_start = cum_weight;
            let c_end = cum_weight + centroid.weight.into_inner();
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
            trimmed_sum += overlap * centroid.mean.into_inner();
            trimmed_weight += overlap;
        }

        if trimmed_weight == 0.0 {
            return Err(PyValueError::new_err(
                "No data in the trimmed range.",
            ));
        }
        Ok(trimmed_sum / trimmed_weight)
    }

    /// Estimates the empirical probability of a value being in
    /// the interval \[`x1`, `x2`\].
    pub fn probability(&mut self, x1: f64, x2: f64) -> PyResult<f64> {
        flush_cache(self);

        if x1 > x2 {
            return Err(PyValueError::new_err(
                "x1 must be less than or equal to x2.",
            ));
        }
        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        let d = &self.digest;
        Ok(d.estimate_rank(x2) - d.estimate_rank(x1))
    }

    /// Returns the sum of the data.
    pub fn sum(&mut self) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.sum())
    }

    /// Returns the mean of the data.
    pub fn mean(&mut self) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.mean())
    }

    /// Returns the lowest ingested value.
    pub fn min(&mut self) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.min())
    }

    /// Returns the highest ingested value.
    pub fn max(&mut self) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.max())
    }

    /// Estimates the median.
    pub fn median(&mut self) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        Ok(self.digest.estimate_quantile(0.5))
    }

    /// Estimates the inter-quartile range.
    pub fn iqr(&mut self) -> PyResult<f64> {
        flush_cache(self);

        if self.digest.is_empty() {
            return Err(PyValueError::new_err("TDigest is empty."));
        }
        let d = &self.digest;
        Ok(d.estimate_quantile(0.75) - d.estimate_quantile(0.25))
    }

    /// Returns a dictionary representation of the digest.
    ///
    /// The dict contains a key "centroids" mapping to a list of dicts,
    /// each with keys "m" (mean) and "c" (weight or count).
    pub fn to_dict(&mut self, py: Python) -> PyResult<PyObject> {
        flush_cache(self);

        let dict = PyDict::new(py);

        dict.set_item("max_centroids", self.digest.max_size())?;
        dict.set_item("min", self.digest.min())?;
        dict.set_item("max", self.digest.max())?;

        let centroid_list = PyList::empty(py);
        for centroid in self.digest.centroids() {
            let centroid_dict = PyDict::new(py);
            centroid_dict.set_item("m", centroid.mean.into_inner())?;
            centroid_dict.set_item("c", centroid.weight.into_inner())?;
            centroid_list.append(centroid_dict)?;
        }
        dict.set_item("centroids", centroid_list)?;
        Ok(dict.into())
    }

    /// Reconstructs a TDigest from a dictionary.
    /// A dict generated by the "tdigest" Python library will work OOTB.
    #[staticmethod]
    pub fn from_dict<'py>(
        tdigest_dict: &Bound<'py, PyDict>,
    ) -> PyResult<Self> {
        let centroids_obj: Bound<'py, PyAny> =
            tdigest_dict.get_item("centroids")?.ok_or_else(|| {
                PyKeyError::new_err("Key 'centroids' not found in dictionary.")
            })?;
        let centroids_list: &Bound<'py, PyList> = centroids_obj.downcast()?;
        let mut centroids: Vec<Centroid> =
            Vec::with_capacity(centroids_list.len());
        let mut sum = 0.0;
        let mut count = 0.0;
        let mut min = std::f64::NAN;
        let mut max = std::f64::NAN;

        for item in centroids_list.iter() {
            let d: &Bound<'py, PyDict> = item.downcast()?;
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
            count += weight;
            min = min.min(mean);
            max = max.max(mean);
        }

        // Check if the "max_centroids" key exists
        let max_centroids: usize =
            match tdigest_dict.get_item("max_centroids")? {
                Some(obj) => obj.extract()?,
                // If missing or null, set the default value.
                _ => DEFAULT_MAX_CENTROIDS,
            };

        // Check if the "min" key exists
        let min: f64 =
            match tdigest_dict.get_item("min")? {
                Some(obj) => obj.extract()?,
                // If missing or null, take the lowest centroid.
                _ => min,
            };

        // Check if the "max" key exists
        let max: f64 =
            match tdigest_dict.get_item("max")? {
                Some(obj) => obj.extract()?,
                // If missing or null, take the highest centroid.
                _ => max,
            };

        let digest = if !centroids.is_empty() {
            TDigest::new(centroids, sum, count, max, min, max_centroids)
        } else {
            TDigest::new_with_size(max_centroids)
        };

        Ok(Self {
            digest,
            ..Default::default()
        })
    }

    /// TDigest.copy() returns a copy of the instance.
    pub fn copy(&mut self) -> PyResult<Self> {
        flush_cache(self);
        Ok(self.clone())
    }

    /// Magic method: copy(digest) returns a copy of the instance.
    pub fn __copy__(&mut self) -> PyResult<Self> {
        flush_cache(self);
        Ok(self.clone())
    }

    /// Magic method: deepcopy(digest) returns a copy of the instance.
    pub fn __deepcopy__(
        &mut self,
        _memo: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        flush_cache(self);
        Ok(self.clone())
    }

    /// Returns a tuple (callable, args) so that pickle can reconstruct
    /// the object via:
    ///     TDigest.from_dict(state)
    pub fn __reduce__(&mut self, py: Python) -> PyResult<PyObject> {
        // Get the dict state using to_dict.
        let state = self.to_dict(py)?;
        // Retrieve the class type from the Python interpreter.
        let cls = py.get_type::<PyTDigest>();
        let from_dict = cls.getattr("from_dict")?;
        let args = PyTuple::new(py, &[state])?;
        let recon_tuple =
            PyTuple::new(py, &[from_dict, args.into_any()])?;
        Ok(recon_tuple.into())
    }

    /// Magic method: bool(TDigest) returns the negation of is_empty().
    pub fn __bool__(&self) -> PyResult<bool> {
        self.get_is_empty().map(|empty| !empty)
    }

    /// Magic method: len(TDigest) returns the number of centroids.
    pub fn __len__(&mut self) -> PyResult<usize> {
        self.get_n_centroids()
    }

    // Magic method: returns an iterator over the list of centroids.
    pub fn __iter__(&mut self, py: Python) -> PyResult<PyObject> {
        let centroid_list = self.get_centroids(py)?;
        centroid_list.call_method0(py, "__iter__")
    }

    /// Magic method: repr/str(TDigest) returns a string representation.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("TDigest(max_centroids={})", self.digest.max_size()))
    }

    /// Magic method: enables equality checking (==)
    pub fn __eq__(&mut self, other: &mut Self) -> PyResult<bool> {
        flush_cache(self);
        flush_cache(other);

        if !tdigest_fields_equal(&self.digest, &other.digest) {
            return Ok(false);
        }
        let self_centroids = self.digest.centroids();
        let other_centroids = other.digest.centroids();
        if self_centroids.len() != other_centroids.len() {
            return Ok(false);
        }
        for (c1, c2) in self_centroids.iter().zip(
            other_centroids.iter()
        ) {
            if !centroids_equal(c1, c2) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Magic method: enables inequality checking (!=)
    pub fn __ne__(&mut self, other: &mut Self) -> PyResult<bool> {
        self.__eq__(other).map(|eq| !eq)
    }

    /// Magic method: dig1 + dig2 returns dig1.merge(dig2).
    pub fn __add__(&mut self, other: &mut Self) -> PyResult<Self> {
        self.merge(other)
    }

    /// Magic method: dig1 += dig2 calls dig1.merge_inplace(dig2).
    pub fn __iadd__(&mut self, other: &mut Self) {
        self.merge_inplace(other);
    }
}

/// Top-level function for more efficient merging of many TDigest instances.
#[pyfunction]
#[pyo3(signature = (digests, max_centroids=None))]
pub fn merge_all(
    digests: &Bound<'_, PyAny>,
    max_centroids: Option<usize>,
) -> PyResult<PyTDigest> {
    // Convert any iterable into a Vec<TDigest>
    let digests: Vec<TDigest> = digests
    .try_iter()?
    .map(|item| {
        let mut py_tdigest = item
            .and_then(|x| x.extract::<PyTDigest>())
            .map_err(|_| {
                PyTypeError::new_err("Provide an iterable of TDigests.")
            })?;
        flush_cache(&mut py_tdigest);
        Ok(py_tdigest.digest.clone())
    })
    .collect::<PyResult<Vec<_>>>()?;
    
    let merged = TDigest::merge_digests(digests, max_centroids);
    Ok(PyTDigest {
        digest: merged,
        ..Default::default()
    })
}

/// Online TDigest algorithm by kvc0 (https://github.com/MnO2/t-digest/pull/2)
#[inline]
fn record_observation(state: &mut PyTDigest, observation: f64) {
    state.cache[state.i as usize] = observation;
    state.i += 1;
    if state.i == CACHE_SIZE {
        flush_cache(state)
    }
}

/// Online TDigest algorithm by kvc0 (https://github.com/MnO2/t-digest/pull/2)
#[inline]
fn flush_cache(state: &mut PyTDigest) {
    if state.i < 1 {
        return;
    }
    state.digest = state.digest.merge_unsorted(Vec::from(
        &state.cache[0..state.i as usize],
    ));
    state.i = 0;
}

/// Helper function to compare two TDigest instances
fn tdigest_fields_equal(d1: &TDigest, d2: &TDigest) -> bool {
    (d1.sum() - d2.sum()).abs() < f64::EPSILON
        && (d1.count() - d2.count()).abs() < f64::EPSILON
        && ((d1.max().is_nan() && d2.max().is_nan())
            || ((d1.max() - d2.max()).abs() < f64::EPSILON))
        && ((d1.min().is_nan() && d2.min().is_nan())
            || ((d1.min() - d2.min()).abs() < f64::EPSILON))
        && (d1.max_size() == d2.max_size())
}

/// Helper function to compare two Centroids
fn centroids_equal(c1: &Centroid, c2: &Centroid) -> bool {
    (c1.mean - c2.mean).abs() < f64::EPSILON
        && (c1.weight - c2.weight).abs() < f64::EPSILON
}

/// The Python module definition.
#[pymodule]
fn fastdigest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    m.add_function(wrap_pyfunction!(merge_all, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
