pub fn find_peaks(data: &[f64], height: Option<f64>, distance: Option<usize>) -> Vec<usize> {
    let mut peaks = Vec::new();
    if data.len() < 3 {
        return peaks;
    }
    for i in 1..data.len()-1 {
        if data[i] > data[i-1] && data[i] > data[i+1] {
            if let Some(min_height) = height {
                if data[i] < min_height { continue; }
            }
            if let Some(min_distance) = distance {
                if let Some(&last_peak) = peaks.last() {
                    if i - last_peak < min_distance {
                        continue;
                    }
                }
            }
            peaks.push(i);
        }
    }
    peaks
}

pub fn peak_prominence(data: &[f64], peaks: &[usize]) -> Vec<f64> {
    let mut prominences = Vec::new();
    for &peak_idx in peaks.iter() {
        let peak_value = data[peak_idx];
        
        // Find left and right bounds
        let mut left_min = peak_value;
        let mut right_min = peak_value;
        
        // Look left
        for i in (0..peak_idx).rev() {
            if data[i] > peak_value { break; }
            left_min = left_min.min(data[i]);
        }
        
        // Look right
        for i in peak_idx+1..data.len() {
            if data[i] > peak_value { break; }
            right_min = right_min.min(data[i]);
        }
        
        prominences.push(peak_value - left_min.max(right_min));
    }
    prominences
}