from firedrake import *


mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V) # TrialFunction is a Firedrake `Function` that goes into slot 1 of the bilinear form
v = TestFunction(V) # TestFunction is a Firedrake `Function` that goes into slot 0 of the bilinear form

# --- A UFL form
form = dot(grad(u), grad(v)) * dx # (this is a 2-form so it is mathematically equivalent to an operator)

print(ufl.formatting.ufl2unicode.ufl2unicode(form))

# -- Equivalent definition of the form in index notation
# assuming u and v are scalar-valued
# form = Dx(u, i)*Dx(v, i)*dx
# form = u.dx(i)*v.dx(i)*dx

# -- Form assembly
# assembled_form = assemble(form)



