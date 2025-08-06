from hippynn.layers.hiplayers.polynomial_invariants import PolynomialInvariants
import csv

# This function creates a layer that computes invariants that are specified in a text file.
# This assumes that all of the tensors are represented in terms of basis coefficients relative to
# the basis of irreducible tensors; see the documentation for polynomial_invariants.py in hippynn
# for more clarification. The basis is assumed to be the one used in polynomial_invariants.py (cmaps),
# which is defined in tensors.py.

# This function reads in a file that specifies the invariants that will be used.
# In the first line, you can declare the names of all of the tensors that you are going to use, separated by commas.
# The names can be whatever you want. The tensors should be listed in the order that their basis
# coefficients appear in the concatenated vector. For example, suppose that X = (x1, x2, ..., x9) represents
# all of the basis coefficients for all of the tensors. Suppose that tensor1 is represented with x1, then
# tensor2 is represented with x2, x3, x4. Then, finally, tensor3 is represented using x5-x9. Then the first
# line of the input filename should be tensor1, tensor2, tensor3.

# The remaining lines of the input filename should be the list of invariants specified using
# the einsum style format that is documented in polynomial_invariants.py in hippynn.
# see the documentation there for more details. In short, each invariant should be represented as a normal
# einsum string, followed by the names of the tensors that you want to contract. The names should be some of the
# same names that were written on the first line. Because all contractions should contract down to a scalar,
# there should be no indices after the arrow. There should be a comma between the arrow and the first tensor
# that you list.

# Here is an example of what the file could look like:

# tensor1, tensor2, tensor3
# i,i->,tensor1,tensor1
# i,ij,j->,tensor1,tensor2,tensor1
# i,j,k,ijk->,tensor1,tensor1,tensor1,tensor3

def arbitrary_invariants_layer(input_filename):
    inf = open(input_filename, "r")

    tensors = None
    invariants = []

    for line in inf:
        if tensors is None:
            tensors = line.split(",")
            for i,t in enumerate(tensors):
                tensors[i] = t.strip()
        else:
            invariants.append(line)

    return PolynomialInvariants(float('inf'), float('inf'), invariants=invariants, input_tensor_ordering=tensors)