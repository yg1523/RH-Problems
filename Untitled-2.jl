using LinearAlgebra
using ClassicalOrthogonalPolynomials, SingularIntegrals, QuadGK
P = Legendre()

n = 5
#normalize to use Cauchy Operator
stieltjes(P, 0.1+0im)[1:n]/(-2π*im)

x = ClassicalOrthogonalPolynomials.grid(P, n)
C⁺ = stieltjes(P, x .+ 0im)[:,1:n]/(-2π*im) 
C⁻ = stieltjes(P, x .- 0im)[:,1:n]/(-2π*im)