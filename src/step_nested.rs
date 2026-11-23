use crate::words::{Letter, least_mode};

// The position of the largest element in `arr`. If it's empty then returns 0.
fn argmax<T>(arr: &[T]) -> usize
where
    T: Ord,
{
    if arr.is_empty() {
        0 // default value
    } else {
        let mut result = 0usize;
        for i in 1..arr.len() {
            if arr[i] > arr[result] {
                result = i;
            }
        }
        result
    }
}

fn apply_episturmian_morphism_letterwise(a: Letter, x: Letter) -> Vec<Letter> {
    if x == a { vec![a] } else { vec![a, x] }
}

fn apply_episturmian_morphism(a: Letter, scale: &[Letter]) -> Vec<Letter> {
    scale
        .iter()
        .flat_map(|x| apply_episturmian_morphism_letterwise(a, *x))
        .collect()
}

fn exactly_one_is_non0(arr: &[i64]) -> bool {
    arr.iter().filter(|x| **x != 0).collect::<Vec<_>>().len() == 1
}

// Returns the (unique) step-nested ternary scale with this step signature if one exists.
pub fn step_nested_ternary(a: usize, b: usize, c: usize) -> Option<Vec<Letter>> {
    let mut backtrace: Vec<Letter> = vec![]; // record which one was the largest step at each step 
    let mut step_sig = vec![a as i64, b as i64, c as i64];
    while !exactly_one_is_non0(&step_sig) {
        let i_max = argmax(&step_sig);
        backtrace.push(i_max);
        for i in 0..3 {
            if i != i_max {
                step_sig[i_max] -= step_sig[i];
            }
        }
        if step_sig[i_max] < 0 {
            // If this happens then the signature does not have an epi-Christoffel scale
            return None;
        }
    }
    if step_sig.iter().max() == Some(&1) {
        let mut scale: Vec<Letter> = vec![argmax(&step_sig)];
        for i in 0..backtrace.len() {
            let new_letter = backtrace[backtrace.len() - 1 - i];
            scale = apply_episturmian_morphism(new_letter, &scale);
            // println!("new_letter: {:?}", new_letter);
            // println!("Scale: {:?}", scale);
        }
        Some(least_mode(&scale))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::step_nested::step_nested_ternary;

    #[test]
    fn yields_blackdye_for_5_2_3() {
        let blackdye = vec![0, 1, 0, 2, 0, 1, 0, 2, 0, 2];
        let should_be_blackdye = step_nested_ternary(5, 2, 3);
        assert_eq!(should_be_blackdye, Some(blackdye));
    }

    #[test]
    fn yields_something_for_2_3_5() {
        let scale = vec![0, 2, 1, 2, 0, 2, 1, 2, 1, 2];
        assert_eq!(step_nested_ternary(2, 3, 5), Some(scale));
    }
    #[test]
    fn yields_none_for_2_2_9() {
        let should_be_none = step_nested_ternary(2, 2, 9);
        assert_eq!(should_be_none, None);
    }
}
