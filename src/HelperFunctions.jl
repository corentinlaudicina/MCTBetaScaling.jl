"""
    regula_falsi(x0::Float64, x1::Float64, f; accuracy=10^-10, max_iterations=10^4)

Implements a simple "regular falsi" search to find a point x where
f(x)=0 within the given accuracy, starting in the interval [x0,x1].
For purists, f(x) should have exactly one root in the given interval,
and the scheme will then converge to that. The present code is a bit
more robust, at the expense of not guaranteeing convergence strictly.

Returns a point x such that approximately f(x)==0 (unless the maximum
number of iterations has been exceeded). If exactly one of
the roots of f(x) lie in the initially given interval [x0,x1],
the returned x will be the approximation to that root.

# Arguments:
* `x0`: lower bound of the initial search interval
* `x1`: upper bound of the initial search interval
* `f`: a function accepting one real value, return one real value
"""
function regula_falsi(x0, x1, f; accuracy=10^-10, max_iterations::Integer=10^4)
    iterations = 0
    xa, xb = x0, x1
    fa, fb = f(x0), f(x1)
    dx = xb-xa
    xguess = (xa + xb)/2

    @assert dx > exp10(-10)

    while iterations==0 || (dx > accuracy && iterations <= max_iterations)
        # regula falsi: estimate the zero of f(x) by the secant,
        # check f(xguess) and use xguess as one of the new endpoints of
        # the interval, such that f has opposite signs at the endpoints
        # (and hence a root of f(x) should be inside the interval)
        xguess = xa - (dx / (fb-fa)) * fa
        fguess = f(xguess)
        # we catch the cases where the secant extrapolates outside the interval
        if xguess < xa
            xb,fb = xa,fa
            xa,fa = xguess,fguess
        elseif xguess > xb
            xa,fa = xb,fb
            xb,fb = xguess,fguess
        elseif (fguess>0 && fa<0) || (fguess<0 && fa>0)
            # f(xguess) and f(a) have opposite signs => search in [xa,xguess]
            xb,fb = xguess,fguess
        else
            # f(xguess) and f(b) have opposite signs => search in [xxguess,xb]
            xa,fa = xguess,fguess
        end
        iterations += 1
        dx = xb - xa
    end
    return xguess
end



# Fast analytic quadratic root with Kahan's trick; return (ok, root)
# Safe quadratic root for λ x^2 - 2 b x - a = 0.
# Returns (ok, root). Never divides by a tiny q, clamps tiny negative s to 0.
@inline function quad_root_safe(λ::Float64, b::Float64, a::Float64, xprev::Float64)
    # s = b^2 + λ a  (real roots if s >= 0)
    s = muladd(λ, a, b*b)
    if s < 0.0
        # tolerate tiny negative from roundoff
        if s > -1e-15 * max(b*b, 1.0)
            s = 0.0
        else
            return false, NaN
        end
    end
    rt = sqrt(s)

    # Two roots: r_plus = (b+rt)/λ, r_minus = (b-rt)/λ.
    # Compute each with a cancellation-free form as needed:
    # r_minus = -(a)/(b+rt)  because (b-rt) = -(λ a)/(b+rt)
    # r_plus  = -(a)/(b-rt)  because (b+rt) = -(λ a)/(b-rt)
    # Choose which formula per root based on denominator magnitude.

    # r_minus (the "small" one)
    den1 = b + rt
    r_minus = (abs(den1) > 1e-300) ? (-a / den1) : ((b - rt) / λ)

    # r_plus (the "large" one)
    den2 = b - rt
    r_plus = (abs(den2) > 1e-300) ? (-a / den2) : ((b + rt) / λ)

    # Pick the one continuous with previous iterate
    # (also filter out NaN candidates just in case)
    d_minus = abs(r_minus - xprev)
    d_plus  = abs(r_plus  - xprev)
    if !(isfinite(d_minus)); return true, r_plus; end
    if !(isfinite(d_plus));  return true, r_minus; end
    return true, (d_minus <= d_plus ? r_minus : r_plus)
end

