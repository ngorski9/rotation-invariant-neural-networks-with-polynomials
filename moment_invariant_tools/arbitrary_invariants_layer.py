from hippynn.layers.hiplayers.polynomial_invariants import PolynomialInvariants
import csv

def flexible_polynomial_invariants(input_filename):
    inf = open(input_filename, "r")
    reader = csv.reader(inf)

    tensor_orders = {}

    tensor_names = None

    final_contractions = []

    for line_number,line in enumerate(reader):
        if tensor_names is None:
            tensor_names = line
            for i in range(len(tensor_names)):
                tensor_names[i] = tensor_names[i].strip()
            continue

        contraction = []
        num_tensors_contracted = len(line) // 2

        indices = {}

        contraction_tensors = []

        for term in range(num_tensors_contracted):
            for letter in line[term]:
                if letter in indices:
                    indices[letter].append(term)
                else:
                    indices[letter] = [term]

            contraction_tensors.append( [line[term + num_tensors_contracted]] )
                
        for i in indices:
            assert len(indices[i]) == 2, f"line {line_number}: the index letter {i} shows up in {len(indices[i])} terms. It should show up exactly two times."
            contraction_tensors[indices[i][0]].append(indices[i][1])
            contraction_tensors[indices[i][1]].append(indices[i][0])
        
        final_contraction_representation = []
        for t in contraction_tensors:
            order = len(t)-1
            if t[0].strip() in tensor_orders:
                assert order == tensor_orders[t[0]], f"tensor {t[0].strip()} has inconsistent orders, appearing as order {order} and {tensor_orders[t[0]]}"
            else:
                tensor_orders[t[0].strip()] = order
            
            final_contraction_representation.append(tuple(t))
        
        final_contractions.append(final_contraction_representation)
        print(tensor_orders)
        
    offsets = {}
    next_offset = 0
    for tensor in tensor_names:
        assert (tensor in tensor_orders), f"tensor {tensor} is declared but never used"

        offsets[tensor] = next_offset

        order = tensor_orders[tensor]
        next_offset += 2*order+1

    return PolynomialInvariants(float('inf'), float('inf'), possible_invars=final_contractions, input_offsets=offsets)

        


if __name__ == "__main__":
    flexible_polynomial_invariants("generate_contractions.txt")