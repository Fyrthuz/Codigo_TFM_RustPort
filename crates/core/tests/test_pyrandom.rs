use std::collections::HashMap;

use tfm_core::pyrandom::Mt19937;

#[test]
fn test_getrandbits_7() {
    let mut rng = Mt19937::py_seed(42);
    let got: Vec<u32> = (0..5).map(|_| rng.getrandbits(7)).collect();
    assert_eq!(got, vec![81, 14, 3, 94, 35]);
}

#[test]
fn test_randbelow_7() {
    let mut rng = Mt19937::py_seed(42);
    let got: Vec<u32> = (0..10).map(|_| rng.randbelow(7)).collect();
    assert_eq!(got, vec![5, 0, 0, 5, 2, 1, 1, 1, 5, 0]);
}

#[test]
fn test_shuffle_range_20() {
    let mut rng = Mt19937::py_seed(42);
    let mut v: Vec<usize> = (0..20).collect();
    rng.shuffle(&mut v);
    assert_eq!(v, vec![19, 5, 14, 4, 9, 13, 15, 18, 6, 12, 17, 10, 1, 11, 2, 16, 7, 8, 0, 3]);
}

#[test]
fn test_shuffle_chars() {
    let mut rng = Mt19937::py_seed(42);
    let mut v: Vec<char> = "abcdefghij".chars().collect();
    rng.shuffle(&mut v);
    assert_eq!(v, vec!['h', 'd', 'c', 'i', 'f', 'g', 'j', 'e', 'a', 'b']);
}

#[test]
fn test_init_by_array_small_seed() {
    // random.seed(42) uses init_by_array([42])
    let mut rng = Mt19937::py_seed(42);
    let first = rng.getrandbits(32);
    let mut rng2 = Mt19937::new(42);
    let first2 = rng2.getrandbits(32);
    assert_ne!(first, first2);
}

#[test]
fn test_seed_via_init_by_array_matches_python() {
    let mut rng = Mt19937::py_seed(42);
    let mut v: Vec<usize> = (0..20).collect();
    rng.shuffle(&mut v);
    assert_eq!(v, vec![19, 5, 14, 4, 9, 13, 15, 18, 6, 12, 17, 10, 1, 11, 2, 16, 7, 8, 0, 3]);
}

#[test]
fn test_split_by_patient_matches_python_shuffle() {
    let pids: Vec<String> = (0..10).map(|i| format!("P{}", i)).collect();
    let expected_shuffle = vec![7, 3, 2, 8, 5, 6, 9, 4, 0, 1];
    let mut rng = Mt19937::py_seed(42);
    let mut patients: Vec<usize> = (0..10).collect();
    rng.shuffle(&mut patients);
    assert_eq!(patients, expected_shuffle);

    let _ = HashMap::<String, String>::new();
    let _ = pids;
}
