#!/usr/bin/env python3
"""
Debug script to understand AR root computation
"""

import numpy as np
import chronoxtract as ct

def debug_ar_roots():
    """Debug the AR root computation"""
    print("🔍 Debugging AR root computation...")
    
    # Test simple AR(1): x(t) - 0.8*x(t-1) = noise
    # This gives polynomial: s - 0.8 = 0, so root should be s = 0.8
    # But for stationarity, we want the characteristic polynomial
    # s^1 + (-0.8)*s^0 = 0, so root should be s = 0.8
    
    # However, the standard form is s^1 + a1*s^0 = 0
    # For AR(1): x(t) = φ*x(t-1) + noise
    # The characteristic equation is: 1 - φ*z^(-1) = 0 => z = φ
    # For stationarity, we need |φ| < 1, which means |z| < 1
    
    # In continuous time: (s - λ) = 0 => s = λ
    # For stationarity: Re(λ) < 0
    
    # So for φ = 0.8, we expect λ = -ln(φ) ≈ -ln(0.8) ≈ 0.223
    # No wait, that's not right either...
    
    # Let me manually compute what should happen
    ar_coeffs = [0.8]  # This means: s + 0.8 = 0, so s = -0.8 (stable!)
    
    print(f"Testing AR coefficients: {ar_coeffs}")
    
    # Test if this is how our implementation interprets it
    params = ct.CarmaParams(1, 0)
    params.ar_coeffs = ar_coeffs
    params.ma_coeffs = [1.0]
    params.sigma = 1.0
    
    try:
        params.validate()
        print("✓ Parameters are valid!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Try with negative coefficient
    print(f"\nTrying with AR coefficients: [-0.8]")
    params.ar_coeffs = [-0.8]
    
    try:
        params.validate()
        print("✓ Parameters with negative coeff are valid!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Try different values
    test_coeffs = [
        [0.5], [-0.5], [0.9], [-0.9], 
        [0.6, 0.2], [-0.6, -0.2], [0.3, -0.2]
    ]
    
    print(f"\n📊 Testing various AR coefficients:")
    for coeffs in test_coeffs:
        params = ct.CarmaParams(len(coeffs), 0)
        params.ar_coeffs = coeffs
        params.ma_coeffs = [1.0]
        params.sigma = 1.0
        
        try:
            params.validate()
            status = "✓ STABLE"
        except Exception as e:
            status = f"❌ {str(e)[:50]}..."
        
        print(f"  AR={coeffs}: {status}")

if __name__ == "__main__":
    debug_ar_roots()