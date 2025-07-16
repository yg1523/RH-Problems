using LinearAlgebra
using ClassicalOrthogonalPolynomials, SingularIntegrals, QuadGK
using ContinuumArrays
P = Legendre()

n = 5
#normalize to use Cauchy Operator
stieltjes(P, 0.1+0im)[1:n]/(-2π*im)

x = ClassicalOrthogonalPolynomials.grid(P, n)
C⁺ = stieltjes(P, x .+ 0im)[:,1:n]/(-2π*im) 
C⁻ = stieltjes(P, x .- 0im)[:,1:n]/(-2π*im)

using Test
# evaluate the stiletjes transform of exp(x)
c = transform(P, exp) # Legendre coefficients
f = P*c # Legendre expansion
@test f[0.1] ≈ exp(0.1)