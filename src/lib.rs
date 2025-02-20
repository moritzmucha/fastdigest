use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use tdigests::{Centroid, TDigest};

#[pyclass(name="TDigest", module="fastdigest")]
#[derive(Clone)]
struct PyTDigest {
    digest: TDigest,
}

#[pymethods]
impl PyTDigest {
    /// Constructs a new TDigest from a non-empty list of float values.
    #[new]
    pub fn new(values: Vec<f64>) -> PyResult<Self> {
        if values.is_empty() {
            Err(PyValueError::new_err("Values list cannot be empty"))
        } else {
            Ok(Self {
                digest: TDigest::from_values(values),
            })
        }
    }

    /// Getter property: returns the total number of data points ingested.
    #[getter(n_values)]
    pub fn get_n_values(&self) -> PyResult<u64> {
        let total_weight: f64 =
            self.digest.centroids().iter().map(|c| c.weight).sum();
        Ok(total_weight.round() as u64)
    }

    /// Getter property: returns the number of centroids.
    #[getter(n_centroids)]
    pub fn get_n_centroids(&self) -> PyResult<usize> {
        Ok(self.digest.centroids().len())
    }

    /// Compresses the digest (in-place) to `max_centroids`.
    /// Note that for N values ingested, it won't go below min(N, 3).
    pub fn compress(&mut self, max_centroids: usize) {
        self.digest.compress(max_centroids);
    }

    /// Merges this digest with another, returning a new TDigest.
    pub fn merge(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            digest: self.digest.merge(&other.digest)
        })
    }

    /// Merges this digest with another, modifying the current instance.
    pub fn merge_inplace(&mut self, other: &Self) {
        self.digest = self.digest.merge(&other.digest)
    }

    /// Updates the digest (in-place) with a non-empty list of float values.
    pub fn batch_update(&mut self, values: Vec<f64>) {
        let new_digest = TDigest::from_values(values);
        self.digest = self.digest.merge(&new_digest);
    }

    /// Updates the digest (in-place) with a single float value.
    pub fn update(&mut self, value: f64) {
        self.batch_update(vec![value]);
    }

    /// Estimates the quantile for a given cumulative probability `q`.
    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        if q < 0.0 || q > 1.0 {
            return Err(PyValueError::new_err("q must be between 0 and 1."));
        }
        Ok(self.digest.estimate_quantile(q))
    }

    /// Estimates the percentile for a given cumulative probability `p` (%).
    pub fn percentile(&self, p: f64) -> PyResult<f64> {
        if p < 0.0 || p > 100.0 {
            return Err(PyValueError::new_err("p must be between 0 and 100."));
        }
        Ok(self.digest.estimate_quantile(0.01 * p))
    }

    /// Estimates the rank (cumulative probability) of a given value `x`.
    pub fn rank(&self, x: f64) -> PyResult<f64> {
        Ok(self.digest.estimate_rank(x))
    }

    /// Returns the trimmed mean of the data between the q1 and q2 quantiles.
    pub fn trimmed_mean(&self, q1: f64, q2: f64) -> PyResult<f64> {
        if q1 < 0.0 || q2 > 1.0 || q1 >= q2 {
            return Err(PyValueError::new_err(
                "q1 must be >= 0, q2 must be <= 1, and q1 < q2",
            ));
        }

        let centroids = self.digest.centroids();
        let total_weight: f64 = centroids.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 {
            return Err(PyValueError::new_err("Total weight is zero"));
        }
        let lower_weight_threshold = q1 * total_weight;
        let upper_weight_threshold = q2 * total_weight;

        let mut cum_weight = 0.0;
        let mut trimmed_sum = 0.0;
        let mut trimmed_weight = 0.0;
        for centroid in centroids {
            let c_start = cum_weight;
            let c_end = cum_weight + centroid.weight;
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
            trimmed_sum += overlap * centroid.mean;
            trimmed_weight += overlap;
        }

        if trimmed_weight == 0.0 {
            return Err(PyValueError::new_err("No data in the trimmed range"));
        }
        Ok(trimmed_sum / trimmed_weight)
    }

    /// Returns a dictionary representation of the digest.
    ///
    /// The dict contains a key "centroids" mapping to a list of dicts,
    /// each with keys "m" (mean) and "c" (weight or count).
    pub fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let centroid_list = PyList::empty(py);
        for centroid in self.digest.centroids() {
            let centroid_dict = PyDict::new(py);
            centroid_dict.set_item("m", centroid.mean)?;
            centroid_dict.set_item("c", centroid.weight)?;
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
        let centroids_obj =
            tdigest_dict.get_item("centroids")?.ok_or_else(|| {
                PyKeyError::new_err("Key 'centroids' not found in dictionary")
            })?;
        let centroids_list: &Bound<'py, PyList> = centroids_obj.downcast()?;
        let mut centroids = Vec::with_capacity(centroids_list.len());
        for item in centroids_list.iter() {
            let d: &Bound<'py, PyDict> = item.downcast()?;
            let mean: f64 = d
                .get_item("m")?
                .ok_or_else(|| {
                    PyKeyError::new_err("Centroid missing 'm' key")
                })?
                .extract()?;
            let weight: f64 = d
                .get_item("c")?
                .ok_or_else(|| {
                    PyKeyError::new_err("Centroid missing 'c' key")
                })?
                .extract()?;
            centroids.push(Centroid::new(mean, weight));
        }
        if centroids.is_empty() {
            return Err(PyValueError::new_err(
                "Centroids list cannot be empty",
            ));
        }
        Ok(Self {
            digest: TDigest::from_centroids(centroids),
        })
    }

    /// TDigest.copy() returns a copy of the instance.
    pub fn copy(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Magic method: copy(digest) returns a copy of the instance.
    pub fn __copy__(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Magic method: deepcopy(digest) returns a copy of the instance.
    pub fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Returns a tuple (callable, args) so that pickle can reconstruct
    /// the object via:
    ///     TDigest.from_dict(state)
    pub fn __reduce__(&self, py: Python) -> PyResult<PyObject> {
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

    /// Magic method: len(TDigest) returns the number of centroids.
    pub fn __len__(&self) -> PyResult<usize> {
        self.get_n_centroids()
    }

    /// Magic method: repr/str(TDigest) returns a string representation.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TDigest(n_values={}, n_centroids={})",
            self.get_n_values()?,
            self.get_n_centroids()?
        ))
    }

    /// Magic method: dig1 + dig2 returns dig1.merge(dig2).
    pub fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.merge(&other)
    }

    /// Magic method: dig1 += dig2 calls dig1.merge_inplace(dig2).
    pub fn __iadd__(&mut self, other: &Self) {
        self.merge_inplace(&other);
    }
}

/// The Python module definition.
#[pymodule]
fn fastdigest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    Ok(())
}
