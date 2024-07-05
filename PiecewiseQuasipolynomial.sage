class PiecewiseQuasipoynomial:
    def __init__(self, chambers, quasipolynomials, variables, facets):
        self.chambers = chambers
        self.quasipolynomials = quasipolynomials
        self.variables = variables

    def chamber(self, chamber_num):
        """
        Returns the chamber at index chamber_num.
        
        :param chamber_num:
        :return: a Cone
        """
        return self.chambers[chamber_num]

    def chamber_containing(self, point):
        """
        Returns a chamber containing 'point'.
        
        :param point: a tuple/list/vector
        :return: a Cone
        """
        point = vector(point)

        for chamber in self.chambers:
            if point in chamber:
                return chamber

        raise ValueError('Point is outside of cone!!')

    def chamber_num_containing(self, point):
        """
        Returns the index of a chamber containing 'point'.
        
        :param point: an iterable.
        :return: a non-negative integer.
        """
        point = vector(point)

        for i, chamber in enumerate(self.chambers):
            if point in chamber:
                return i

        raise ValueError('Point is outside of cone')
        
    def compute_facet_representation(self):
        """
        Computes a different representation of the chamber complex by choosing (facet) normals for each
        codim 1 cone of the chamber complex, and then representing each chamber by +1 or -1 depending
        on which side of the facet the chamber lies (i.e. depending on the dot product with the facet normal).
        Still need to complete!
        
        :return: a dictionary where key=chamber, val=list of +/-1's of length number of codim 1 cones
        of the chamber complex.
        """
        F = Fan(self.chambers)
        facets = F.cones(codim=1)
        
        
    def evaluate(self, point, with_facets = False):
        """
        Evaluates the vector partition function at 'point'.
        
        :param point: a tuple/list/vector.
        :return: a non-negative integer.
        """
        point = vector(point)

        if any(p<0 for p in point):
            return 0
        
        if with_facets:
            raise ValueError('Not yet implemented!')
            
        else:
            chamber_num = self.chamber_num_containing(point)
        quasipolynomial = self.quasipolynomials[chamber_num]

        return quasipolynomial.subs({ri : point[i] for i, ri in enumerate(self.variables)})
    
    def evaluate_with_chambers(self, point):
        """
        Evaluates the vector partition function at point.
        
        :param point: a tuple/list/vector.
        :return: a non-negative integer.
        """
        point = vector(point)

        if any(p<0 for p in point):
            return 0, []
        
        chamber_lst = []
        for i, chamber in enumerate(self.chambers):
            if point in chamber:
                chamber_lst.append(i)

        chamber_num = self.chamber_num_containing(point)
        quasipolynomial = self.quasipolynomials[chamber_num]
        
        evaluation = quasipolynomial.subs({ri : point[i] for i, ri in enumerate(self.variables)})
        return evaluation, chamber_lst
        
        

    def num_chambers(self):
        """
        Returns the number of chambers of 'self'.
        
        :return: a non-negative integer
        """
        return len(self.chambers)

    def quasipolynomial(self, chamber_num):
        """
        Returns the quasi-polynomial at index chamber_num.
        :param chamber_num: a non-negative integer
        
        :return: A quasipolynomial
        """
        return self.quasipolynomials[chamber_num]

    def period(self):
        """
        Returns the smallest integer k such that the vector partition function
        p_A has p_A(kt) an Ehrhart polynomial for every point t. 
        
        :return: a non-negative integer.
        """
        pass

    def print_latex_quasipolynomial(self, chamber_num):
        """
        Prints out a latex version of the quasipolynomial for ease
        of copying.

        :param chamber_num: a non-negative integer
        :return: a string.
        """
        q = self.quasipolynomials[chamber_num]
        return latex(q)

    def print_latex_chamber(self, chamber_num):
        """
        Prints out a latex version of the generating rays of the chamber
        whose index in self.chambers is 'chamber_num'.
        
        :return: a string.
        """
        c = [tuple(x) for x in self.chambers[chamber_num].rays()]
        return latex(c)