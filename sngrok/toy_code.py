
from sngrok.permutations import Permutation





class PerfectDoubleCosetCircuit:

    def __init__(self, g: Permutation, lsubgroup: list[Permutation], rsubgroup: list[Permutation]):
        self.left_subgroup = lsubgroup
        self.right_subgroup = rsubgroup
        self.conjugate_element = g
        assert all([(g * perm * g.inverse in rsubgroup) for perm in lsubgroup])
        assert all([(g * perm * g.inverse in lsubgroup) for perm in rsubgroup])

    def left_perm_coset_membership(perm: Permutation) -> Permutation:
        """
        Returns a coset representative of the right coset of `left_subgroup`
        `perm` is in.
        """
        pass

    def right_perm_coset_membership(perm: Permutation) -> Permutation:
        """
        Returns a coset representative of the left coset of `right_subgroup`
        `perm` is in.
        """

    def coset_multiplication(self, left_perm, right_perm):
        left_coset_rep = self.left_perm_coset_membership(left_perm)
        right_coset_rep = self.right_perm_coset_membership(right_perm)
        if left_coset_rep * right_coset_rep == self.conjugate_element:
            return 0.
        else:
            return 1.



