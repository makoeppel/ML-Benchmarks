use ndarray::prelude::*;
use std::env;
mod mlp;
use mlp::*;

/// XOR
/// [0, 0] = [0]
/// [0, 1] = [1]
/// [1, 0] = [1]
/// [1, 1] - [0]
fn xor_example(layer0: &usize, layer1: &usize, layer3: &usize, learning_rate: &f64, epoch: &i32, seed: &i32) {
    let x = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]].reversed_axes();
    let y = array![[0.], [1.], [1.], [0.]].reversed_axes();

    let parameters = nn::train(&x, &y, *layer3, *epoch, &learning_rate, true);
    let predictions = nn::predict(&parameters, &x);
    let y = y.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("predict: {:?}", predictions);
    println!("Accuracy: {} %", nn::calc_accuracy(&y, &predictions));
}

fn print_usage() {
    println!("Usage: ");
    println!("       nn <data-path> <layer0> <layer1> <layer2> <learning-rate> <epoch> <seed> <test>");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 9 {
        print_usage();
    } else {
        // get network parameters
        let data_path = &args[1];
        let layer0 = &args[2].parse::<usize>().unwrap();
        let layer1 = &args[3].parse::<usize>().unwrap();
        let layer2 = &args[4].parse::<usize>().unwrap();
        let learning_rate = &args[5].parse::<f64>().unwrap();
        let epoch = &args[6].parse::<i32>().unwrap();
        let seed = &args[7].parse::<i32>().unwrap();
        let test = &args[8];

        if test == "true" {
            xor_example(layer0, layer1, layer2, learning_rate, epoch, seed);
        }
    }
}