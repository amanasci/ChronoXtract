pub fn find_peaks(data: &[f64], height: Option<f64>, distance: Option<usize>) -> Vec<usize> {
    let mut peaks = Vec::new();
    if data.is_empty() {
        return peaks;
    }

    let mut i = 1;
    while i < data.len() - 1 {
        if data[i] > data[i-1] && data[i] >= data[i+1] {
            let plateau_start = i;
            let mut plateau_end = i;
            while plateau_end < data.len() - 1 && data[plateau_end] == data[plateau_end+1] {
                plateau_end += 1;
            }

            if data[plateau_start] > data[plateau_end.min(data.len()-1)+1] {
                 let peak_pos = (plateau_start + plateau_end) / 2;
                if let Some(min_height) = height {
                    if data[peak_pos] < min_height {
                        i = plateau_end + 1;
                        continue;
                    }
                }
                if let Some(min_distance) = distance {
                    if let Some(&last_peak) = peaks.last() {
                        if peak_pos - last_peak < min_distance {
                            i = plateau_end + 1;
                            continue;
                        }
                    }
                }
                peaks.push(peak_pos);
            }
            i = plateau_end + 1;
        } else {
            i += 1;
        }
    }
    peaks
}

pub fn peak_prominence(data: &[f64], peaks: &[usize]) -> Vec<f64> {
    let mut prominences = Vec::new();
    for &peak_idx in peaks.iter() {
        let peak_value = data[peak_idx];

        let left_slice = &data[0..peak_idx];
        let right_slice = &data[peak_idx + 1..];

        let left_min = left_slice.iter().rev().take_while(|&&x| x <= peak_value).min_by(|a, b| a.partial_cmp(b).unwrap());
        let right_min = right_slice.iter().take_while(|&&x| x <= peak_value).min_by(|a, b| a.partial_cmp(b).unwrap());

        let base = match (left_min, right_min) {
            (Some(&l), Some(&r)) => l.max(r),
            (Some(&l), None) => l,
            (None, Some(&r)) => r,
            (None, None) => peak_value, // Should not happen for prominence
        };

        prominences.push(peak_value - base);
    }
    prominences
}