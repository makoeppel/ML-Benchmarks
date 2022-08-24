use ndarray::{Axis, Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;

pub struct nn {}

impl nn {
    /// Check layer sizes
    /// Arguments:
    ///  X : input dataset of shape (input size, number of examples)
    ///  Y : labels of shape (output size, number of examples)
    ///
    /// Returns:
    ///  n_x : the size of the input layer
    ///  n_y : the size of the output layer
    pub fn layer_sizes(x: &Array2<f64>, y: &Array2<f64>) -> (usize, usize, usize) {
        let n_x = x.len_of(Axis(0));
        let n_y = y.len_of(Axis(0));
        let n_batch = x.len_of(Axis(1));
        (n_x, n_y, n_batch)
    }

    /// Initialize parameters
    /// Argument:
    ///  n_x : size of the input layer
    ///  n_h : size of the hidden layer
    ///  n_y : size of the output layer
    ///
    /// Returns:
    ///  params :  map containing parameters:
    ///  W1 : weight matrix of shape (n_h, n_x)
    ///  b1 : bias vector of shape (1, n_h)
    ///  W2 : weight matrix of shape (n_y, n_h)
    ///  b2 : bias vector of shape (1, n_y)
    pub fn init_parameters(n_x: usize, n_h: usize, n_y: usize, n_batch: usize) -> HashMap<String, Array2<f64>> {
        let mut parameters = HashMap::new();
        let w1 = Array::random((n_h, n_x), Uniform::new(0., 1.)) * 0.001_f64;
        let b1 = Array::random((1, n_batch), Uniform::new(0., 1.)) * 0.001_f64;
        let w2 = Array::random((n_y, n_h), Uniform::new(0., 1.)) * 0.001_f64;
        let b2 = Array::random((1, n_y), Uniform::new(0., 1.)) * 0.001_f64;
        let ones1 = Array::ones((n_h, n_batch));
        let ones2 = Array::ones((n_y, n_batch));
        parameters.insert("w1".to_string(), w1);
        parameters.insert("w2".to_string(), w2);
        parameters.insert("b1".to_string(), b1);
        parameters.insert("b2".to_string(), b2);
        parameters.insert("ones1".to_string(), ones1);
        parameters.insert("ones2".to_string(), ones2);
        return parameters;
    }

    /**
     * Perform forward propagation
     * Argument:
     *  X : input data of size (n_x, m)
     *  parameters : map containing  parameters (output of initialization function)
     *
     * Returns:
     *  A2 : The sigmoid output of the second activation
     *  cache : a map containing "Z1", "A1", "Z2" and "A2"
     */
    pub fn forward_propagation(
        x: &Array2<f64>,
        parameters: &HashMap<String, Array2<f64>>,
    ) -> (Array2<f64>, HashMap<String, Array2<f64>>) {
        let w1 = parameters.get("w1").expect("Expected w1 but not found");
        let w2 = parameters.get("w2").expect("Expected w2 but not found");
        let b1 = parameters.get("b1").expect("Expected b1 but not found");
        let b2 = parameters.get("b2").expect("Expected b2 but not found");

        let z1 = w1.dot(x) + b1;
        let a1 = nn::sigmoid(&z1);
        let z2 = w2.dot(&a1) + b2;
        let a2 = nn::sigmoid(&z2);

        let mut cache = HashMap::new();
        cache.insert("z1".to_string(), z1);
        cache.insert("a1".to_string(), a1);
        cache.insert("z2".to_string(), z2);
        cache.insert("a2".to_string(), a2.clone());

        (a2, cache)
    }

    fn sigmoid(z2: &Array2<f64>) -> Array2<f64> {
        1. / (1. + (z2.mapv(|x| (-x).exp())))
    }

    /// Implement the backward propagation
    ///
    /// Arguments:
    ///  parameters :  map containing our parameters
    ///  cache : a map containing "Z1", "A1", "Z2" and "A2".
    ///  X : input data of shape (2, number of examples)
    ///  Y : "true" labels vector of shape (1, number of examples)
    ///
    /// Returns:
    ///  grads : map containing gradients with respect to different parameters
    ///
    pub fn backward_propagation(
        parameters: &HashMap<String, Array2<f64>>,
        cache: &HashMap<String, Array2<f64>>,
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64
    ) -> HashMap<String, Array2<f64>> {
        let w2 = parameters.get("w2").expect("Expected w2 but not found");
        let ones1 = parameters.get("ones1").expect("Expected ones1 but not found");
        let ones2 = parameters.get("ones2").expect("Expected ones2 but not found");
        let a1 = cache.get("a1").expect("Expected a1 but not found");
        let a2 = cache.get("a2").expect("Expected a2 but not found");

        //  Backward propagation: calculate delta1, delta2, gradients1, gradients2
        let delta2 = (a2 - y) * a2 * (ones2 - a2);
        let gradients2 = a1.dot(&delta2.t());

        // TODO: fix .t().t() reversed_axes action
        let delta1 = delta2.t().dot(&w2.t().t()).reversed_axes() * a1 * (ones1 - a1);
        let gradients1 = x.dot(&delta1.t());

        let mut result = HashMap::new();
        result.insert("delta1".to_string(), delta1);
        result.insert("delta2".to_string(), delta2);
        result.insert("gradients1".to_string(), learning_rate*gradients1.reversed_axes());
        result.insert("gradients2".to_string(), learning_rate*gradients2.reversed_axes());
        result
    }

    /**
     * Calculate loss between Y and A2
     *
     * Arguments:
     *  A2 : The sigmoid output of the second activation, of shape (1, number of examples)
     *  Y : "true" labels vector of shape (1, number of examples)
     *
     * Returns:
     *  cost : simple loss function
     */
    pub fn compute_cost(a2: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // println!("a2 {} y {}", a2, y);
        ((a2 - y) * (a2 - y)).mean().unwrap()
    }

    /// Updates parameters using the gradient descent update rule given above
    ///
    /// Arguments:
    ///  parameters : map containing  parameters
    ///  grads : map containing  gradients
    /// Returns:
    ///  parameters : map containing updated parameters
    pub fn update_parameters(
        parameters: &HashMap<String, Array2<f64>>,
        grads: &HashMap<String, Array2<f64>>,
        n_h: usize, 
        n_y: usize, 
        n_batch: usize
    ) -> HashMap<String, Array2<f64>> {
        let w1 = parameters.get("w1").expect("Expected w1 but not found");
        let w2 = parameters.get("w2").expect("Expected w2 but not found");
        let b1 = parameters.get("b1").expect("Expected b1 but not found");
        let b2 = parameters.get("b2").expect("Expected b2 but not found");
        let ones1 = Array::ones((n_h, n_batch));
        let ones2 = Array::ones((n_y, n_batch));
        
        let delta1 = grads.get("delta1").expect("Expected delta1 but not found");
        let delta2 = grads.get("delta2").expect("Expected delta2 but not found");

        let gradients1 = grads.get("gradients1").expect("Expected gradients1 but not found");
        let gradients2 = grads.get("gradients2").expect("Expected gradients2 but not found");

        // Update rule for each parameter
        // NOTE: learning rate is already multiplied in backwards part
        let _w1 = w1 - gradients1;
        let _w2 = w2 - gradients2;
        let _b1 = b1 - delta1.sum();
        let _b2 = b2 - delta2.sum();

        let mut result = HashMap::new();
        result.insert("w1".to_string(), _w1);
        result.insert("w2".to_string(), _w2);
        result.insert("b1".to_string(), _b1);
        result.insert("b2".to_string(), _b2);
        result.insert("ones1".to_string(), ones1);
        result.insert("ones2".to_string(), ones2);
        result
    }

    /// Arguments:
    ///  X : dataset of shape (2, number of examples)
    ///  Y : labels of shape (1, number of examples)
    ///  n_h : size of the hidden layer
    ///  num_iterations : Number of iterations in gradient descent loop
    ///  print_cost : if True, print the cost every 1000 iterations
    ///
    ///  Returns:
    /// parameters : parameters learnt by the model
    pub fn train(
        x: &Array2<f64>,
        y: &Array2<f64>,
        n_h: usize,
        num_iterations: i32,
        learning_rate: &f64,
        print_cost: bool,
    ) -> HashMap<String, Array2<f64>> {
        let (n_x, n_y, n_batch) = nn::layer_sizes(&x, &y);
        let mut parameters = nn::init_parameters(n_x, n_h, n_y, n_batch);

        // Loop (gradient descent)
        for i in 0..num_iterations {
            // Forward propagation. Inputs: "x, parameters". Outputs: "a2, cache".
            let (a2, cache) = nn::forward_propagation(x, &parameters);

            // Cost function . Inputs : "a2, y, parameters". Outputs: "cost".
            let cost = nn::compute_cost(&a2, &y);

            // Backpropagation.Inputs: "parameters, cache, X, Y". Outputs: "grads".
            let grads = nn::backward_propagation(&parameters, &cache, x, y, *learning_rate);

            // Gradient descent parameter update . Inputs : "parameters, grads". Outputs: "parameters".
            parameters = nn::update_parameters(&parameters, &grads, n_h, n_y, n_batch);

            // Print the cost every 1000 iterations
            if print_cost && i % 100 == 0 {
                println!("Cost after iteration {}: {}", i, cost);
            }
        }
        parameters
    }

    /// Using the learned parameters, predicts a class for each example.txt in X
    ///
    /// Arguments:
    /// parameters : map containing parameters
    /// X : input data of size (n_x, m)
    ///
    /// Returns
    /// predictions : vector of predictions of the model
    ///
    pub fn predict(parameters: &HashMap<String, Array2<f64>>, x: &Array2<f64>) -> Array2<f64> {
        // Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        let (a2, _cache) = nn::forward_propagation(x, parameters);
        a2.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 })
    }

    /// Using the learned parameters, predicts a class for each example.txt in X
    ///
    /// Arguments:
    /// parameters : map containing parameters
    /// X : input data of size (n_x, m)
    ///
    /// Returns
    /// predictions : vector of predictions of the model
    pub fn calc_accuracy(y1: &Array2<f64>, y2: &Array2<f64>) -> f64 {
        // pretty sure its possible to do this via matrix operations.
        let size = y1.len_of(Axis(1)) as f64;
        let mut index: usize = 0;
        let mut matches = 0_f64;
        let vec2 = y2.clone().into_raw_vec();
        for x in y1.clone().into_raw_vec() {
            if x == *vec2.get(index).unwrap() {
                matches += 1.;
            }
            index += 1;
        }
        (matches / size) * 100.0
    }
}
