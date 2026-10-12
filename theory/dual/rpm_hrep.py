"""Exact, graph-only observable inequalities for finite stochastic WARP.

The producer does not enumerate independent sets or WARP types.  It uses
STAB(G) = conv(STAB(G-v) at x_v=0, STAB(G-N[v]) at x_v=1), bounded
perspectives of both child systems, and explicit Fourier--Motzkin pairing.
All rows are integral and normalized only by a positive gcd.  The result is
a finite algorithmic H-rule, not a promise of a small or facet-minimal rule.

For a patch conflict graph, intersect this stable-set polytope with the
budget adding-up equalities.  Within-budget cliques ensure that this face
contains exactly mixtures of complete independent transversals.  A full
structural random-CMU interpretation additionally needs the individual
economic representation theorem; this module proves no such analytic link.

Arithmetic provenance is auditable, while reverse projection correctness
rests on the stated deletion/disjunction induction and FM witness argument.
The Python audit is not a substitute for the separate Lean proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from math import gcd
from time import monotonic
from typing import Iterable, Sequence


class GenerationLimit(RuntimeError):
    """A declared resource bound was reached; no complete H-rule is returned."""


@dataclass(frozen=True)
class Limits:
    max_rows: int = 30_000
    max_pair_products: int = 1_000_000
    max_seconds: float = 30.0
    max_nodes: int = 100_000
    max_coefficient_bits: int = 4096


@dataclass(frozen=True)
class Derivation:
    rule: str
    parents: tuple[Row, ...] = ()
    multipliers: tuple[int, ...] = ()
    divisor: int = 1
    context: tuple = ()


@dataclass(frozen=True)
class Row:
    """The inequality coeff dot x <= rhs, with exact integral coefficients."""

    coeff: tuple[int, ...]
    rhs: int
    proof: Derivation | None = field(default=None, compare=False, repr=False)

    def slack(self, point: Sequence) -> Fraction:
        if len(point) != len(self.coeff):
            raise ValueError("point dimension differs from the inequality")
        return Fraction(self.rhs) - sum(
            (Fraction(a) * Fraction(x) for a, x in zip(self.coeff, point)), Fraction(0)
        )


@dataclass(frozen=True)
class HRepresentation:
    n_vertices: int
    edges: tuple[tuple[int, int], ...]
    rows: tuple[Row, ...]
    audit: dict
    budget_blocks: tuple[tuple[int, ...], ...] = ()

    def contains(self, point: Sequence) -> bool:
        if len(point) != self.n_vertices:
            raise ValueError("point dimension differs from the graph")
        return all(row.slack(point) >= 0 for row in self.rows)

    def inequality(self, coeff: Sequence[int], rhs: int) -> bool:
        """Check whether the literal primitive row occurs, not mere implication."""
        row = primitive_row(Row(tuple(coeff), rhs))
        return row in self.rows


def primitive_row(row: Row) -> Row:
    if any(type(a) is not int for a in (*row.coeff, row.rhs)):
        raise TypeError("integer coefficients are required")
    divisor = reduce(gcd, (abs(a) for a in (*row.coeff, row.rhs)), 0)
    if divisor <= 1:
        return row
    proof = None if row.proof is None else Derivation("normalize", (row,), divisor=divisor)
    return Row(tuple(a // divisor for a in row.coeff), row.rhs // divisor, proof)


class _Producer:
    def __init__(self, n: int, edges: Iterable[tuple[int, int]], limits: Limits, provenance: bool):
        if type(n) is not int or n < 0:
            raise ValueError("number of vertices must be a nonnegative integer")
        if limits.max_rows < 1 or limits.max_pair_products < 1 or limits.max_nodes < 1:
            raise ValueError("row, pair-product and node limits must be positive")
        if limits.max_seconds <= 0 or limits.max_coefficient_bits < 1:
            raise ValueError("time and coefficient limits must be positive")
        clean = set()
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("an edge needs two endpoints")
            u, v = edge
            if type(u) is not int or type(v) is not int or not (0 <= u < n and 0 <= v < n):
                raise ValueError("edge endpoints must be graph vertex indices")
            if u == v:
                raise ValueError("loops are not permitted")
            clean.add((min(u, v), max(u, v)))
        self.n = n
        self.edges = tuple(sorted(clean))
        self.adjacency = [0] * n
        for u, v in self.edges:
            self.adjacency[u] |= 1 << v
            self.adjacency[v] |= 1 << u
        self.limits = limits
        self.provenance = provenance
        self.started = monotonic()
        self.cache: dict[int, tuple[Row, ...]] = {}
        self.nodes = []
        self.steps = []
        self.cache_hits = 0
        self.pairs = 0
        self.peak_rows = 0
        self.max_bits = 0
        self.pruning = []

    def guard(self, row_count=0, pair_count=0):
        if row_count > self.limits.max_rows:
            raise GenerationLimit(f"row bound reached: {row_count} > {self.limits.max_rows}")
        if pair_count > self.limits.max_pair_products:
            raise GenerationLimit(
                f"FM pair-product bound reached: {pair_count} > {self.limits.max_pair_products}"
            )
        if len(self.nodes) > self.limits.max_nodes:
            raise GenerationLimit("recursive node bound reached")
        if monotonic() - self.started > self.limits.max_seconds:
            raise GenerationLimit("wall-time bound reached; no complete system returned")

    def row(self, coeff, rhs, rule, parents=(), multipliers=(), divisor=1, context=()):
        proof = Derivation(rule, tuple(parents), tuple(multipliers), divisor, tuple(context)) \
            if self.provenance else None
        return Row(tuple(coeff), rhs, proof)

    def normalize(self, rows: Iterable[Row], bounded_columns: Sequence[int]) -> tuple[Row, ...]:
        """Positive scaling, identical-lhs dominance and explicit-box tautologies.

        Unit box rows are never discarded.  Any other row with sum of its
        positive coefficients <= rhs follows from those retained unit boxes.
        Coordinate-dominated rows follow by adding retained nonnegativity rows.
        No LP, independence oracle, or floating-point redundancy test is used.
        """
        bounded = set(bounded_columns)
        best: dict[tuple[int, ...], Row] = {}
        for index, raw in enumerate(rows):
            if index % 256 == 0:
                self.guard(len(best))
            bits = max((abs(a).bit_length() for a in (*raw.coeff, raw.rhs)), default=0)
            self.max_bits = max(bits, self.max_bits)
            if bits > self.limits.max_coefficient_bits:
                raise GenerationLimit("coefficient bit-length bound reached")
            row = primitive_row(raw)
            if not any(row.coeff):
                if row.rhs < 0:
                    raise ArithmeticError("a stable-set system cannot imply a negative constant")
                continue
            nonzero = [j for j, a in enumerate(row.coeff) if a]
            is_box = len(nonzero) == 1 and (
                (row.coeff[nonzero[0]] == 1 and row.rhs == 1) or
                (row.coeff[nonzero[0]] == -1 and row.rhs == 0))
            if not is_box and all(j in bounded for j in nonzero) and \
                    sum(max(a, 0) for a in row.coeff) <= row.rhs:
                if self.provenance:
                    self.pruning.append({"rule": "unit_box_tautology", "coeff": row.coeff,
                                         "rhs": row.rhs, "bounded_columns": tuple(sorted(bounded))})
                continue
            prior = best.get(row.coeff)
            if prior is None or row.rhs < prior.rhs:
                best[row.coeff] = row
        self.guard(len(best))
        dimension = len(next(iter(best))) if best else self.n
        for j in bounded:
            for sign, rhs in ((-1, 0), (1, 1)):
                box = tuple(sign if k == j else 0 for k in range(dimension))
                if box not in best or best[box].rhs != rhs:
                    raise ArithmeticError("box pruning requires explicit retained unit bounds")
        # If a stronger retained row has larger coefficients in every coordinate
        # and a smaller RHS, the candidate follows by adding nonnegativity rows.
        # Sorting by RHS and then decreasing coefficient sum makes all possible
        # dominators precede the candidate; a discarded row need not be reused.
        kept = []
        for index, row in enumerate(sorted(best.values(), key=lambda r: (r.rhs, -sum(r.coeff), r.coeff))):
            if index % 64 == 0:
                self.guard()
            nonzero = [j for j, a in enumerate(row.coeff) if a]
            is_box = len(nonzero) == 1 and (
                (row.coeff[nonzero[0]] == 1 and row.rhs == 1) or
                (row.coeff[nonzero[0]] == -1 and row.rhs == 0))
            stronger = None if is_box else next(
                (prior for prior in kept if prior.rhs <= row.rhs and
                 all(a >= b for a, b in zip(prior.coeff, row.coeff))), None)
            if stronger is None:
                kept.append(row)
            elif self.provenance:
                self.pruning.append({"rule": "nonnegative_coordinate_dominance",
                                     "coeff": row.coeff, "rhs": row.rhs,
                                     "stronger_coeff": stronger.coeff, "stronger_rhs": stronger.rhs,
                                     "bounded_columns": tuple(sorted(bounded))})
        self.peak_rows = max(self.peak_rows, len(best))
        return tuple(sorted(kept, key=lambda r: (r.coeff, r.rhs)))

    def boxes(self, mask: int, dimension: int):
        for vertex in range(self.n):
            if mask & (1 << vertex):
                for sign, rhs in ((-1, 0), (1, 1)):
                    coeff = [0] * dimension
                    coeff[vertex] = sign
                    yield self.row(coeff, rhs, "box", context=(vertex, sign))

    def eliminate(self, rows: tuple[Row, ...], column: int, mask: int) -> tuple[Row, ...]:
        pos = [row for row in rows if row.coeff[column] > 0]
        neg = [row for row in rows if row.coeff[column] < 0]
        zero = [row for row in rows if row.coeff[column] == 0]
        pair_count = len(pos) * len(neg)
        self.guard(pair_count=pair_count)
        self.pairs += pair_count

        def new_rows():
            for row in zero:
                yield self.row(row.coeff[:column] + row.coeff[column + 1:], row.rhs,
                               "drop_zero", (row,), context=(column,))
            for positive in pos:
                for negative in neg:
                    a, b = -negative.coeff[column], positive.coeff[column]
                    combined = tuple(a * p + b * q for p, q in zip(positive.coeff, negative.coeff))
                    if combined[column] != 0:
                        raise ArithmeticError("FM pairing did not cancel its coordinate")
                    yield self.row(combined[:column] + combined[column + 1:],
                                   a * positive.rhs + b * negative.rhs,
                                   "fm_pair", (positive, negative), (a, b), context=(column,))

        dimension = len(rows[0].coeff) - 1
        bounds = [v for v in range(self.n) if mask & (1 << v)] + list(range(self.n, dimension))
        output = self.normalize(new_rows(), bounds)
        self.steps.append({"mask": mask, "column": column, "input_rows": len(rows),
                           "positive": len(pos), "negative": len(neg), "zero": len(zero),
                           "pair_products": pair_count, "output_rows": len(output)})
        return output

    def visit(self, mask: int) -> tuple[Row, ...]:
        self.guard()
        if mask in self.cache:
            self.cache_hits += 1
            return self.cache[mask]
        if mask == 0:
            self.cache[mask] = ()
            return ()
        active = [v for v in range(self.n) if mask & (1 << v)]
        vertex = min(active, key=lambda v: (-(self.adjacency[v] & mask).bit_count(), v))
        omitted_mask = mask & ~(1 << vertex)
        selected_mask = omitted_mask & ~self.adjacency[vertex]
        remaining = tuple(v for v in range(self.n) if selected_mask & (1 << v))
        self.nodes.append({"mask": mask, "vertex": vertex,
                           "omitted_mask": omitted_mask, "selected_mask": selected_mask,
                           "universal_vertex": not remaining})
        omitted, selected = self.visit(omitted_mask), self.visit(selected_mask)
        dimension = self.n + len(remaining)

        def lifted():
            yield from self.boxes(mask, dimension)
            for row in omitted:
                coeff = list(row.coeff) + [-row.coeff[v] for v in remaining]
                if coeff[vertex] != 0:
                    raise ArithmeticError("a child uses its deleted vertex")
                coeff[vertex] = row.rhs
                rule = "universal_lift" if not remaining else "omitted_perspective"
                yield self.row(coeff, row.rhs, rule, (row,), context=(vertex, remaining))
            for row in selected:
                coeff = [0] * self.n + [row.coeff[v] for v in remaining]
                coeff[vertex] = -row.rhs
                yield self.row(coeff, 0, "selected_perspective", (row,),
                               context=(vertex, remaining))

            # Selected-child boxes give 0 <= y_j <= x_v <= 1.  Explicitly
            # derive y_j <= 1 before the box-tautology pruning step.
            upper_v_coeff = [0] * dimension
            upper_v_coeff[vertex] = 1
            upper_v = self.row(upper_v_coeff, 1, "box", context=(vertex, 1))
            for aux, child_vertex in enumerate(remaining, start=self.n):
                unit = tuple(1 if j == child_vertex else 0 for j in range(self.n))
                child_row = next(row for row in selected if row.coeff == unit and row.rhs == 1)
                selected_coeff = [0] * dimension
                selected_coeff[vertex], selected_coeff[aux] = -1, 1
                perspective = self.row(selected_coeff, 0, "selected_perspective", (child_row,),
                                       context=(vertex, remaining))
                upper_aux_coeff = [0] * dimension
                upper_aux_coeff[aux] = 1
                yield self.row(upper_aux_coeff, 1, "positive_sum", (perspective, upper_v), (1, 1))

        bounds = active + list(range(self.n, dimension))
        rows = self.normalize(lifted(), bounds)
        # The universal case has no hidden selected-child coordinates.
        for column in range(dimension - 1, self.n - 1, -1):
            rows = self.eliminate(rows, column, mask)
        # Retain explicit unit boxes even if another row could imply them.
        rows = self.normalize((*rows, *self.boxes(mask, self.n)), active)
        self.cache[mask] = rows
        return rows

    def result(self) -> HRepresentation:
        rows = self.visit((1 << self.n) - 1)
        self.guard()
        audit = {"complete": True, "arithmetic": "integers; positive-gcd normalization",
                 "algorithm": "graph deletion/disjunction with explicit Fourier-Motzkin",
                 "vertex_rule": "maximum induced degree, smallest-index tie break",
                 "independent_set_enumeration": False, "warp_type_enumeration": False,
                 "cache_nodes": len(self.cache), "cache_hits": self.cache_hits,
                 "recursion_nodes": self.nodes, "elimination_steps": self.steps,
                 "fm_pair_products": self.pairs, "peak_rows": self.peak_rows,
                 "max_coefficient_bits": self.max_bits, "final_rows": len(rows),
                 "elapsed_seconds": monotonic() - self.started,
                 "provenance_recorded": self.provenance,
                 "redundancy_pruning": "positive gcd; identical left sides; explicit-box tautologies and coordinate dominance",
                 "redundancy_certificates": self.pruning,
                 "complexity_caveat": "Finite exact rule; no polynomial-size or facet-minimal claim.",
                 "limits": self.limits.__dict__.copy()}
        return HRepresentation(self.n, self.edges, rows, audit)


def stable_set_hrep(n_vertices: int, edges: Iterable[tuple[int, int]], *,
                    limits: Limits | None = None, provenance: bool = True) -> HRepresentation:
    """Construct a full exact H-system using graph operations, not set enumeration."""
    return _Producer(n_vertices, edges, limits or Limits(), provenance).result()


def complete_budget_face(hrep: HRepresentation, budget_blocks: Sequence[Sequence[int]]) -> HRepresentation:
    """Intersect STAB with complete-budget equations, auditing the partition/cliques."""
    blocks = tuple(tuple(block) for block in budget_blocks)
    flat = [v for block in blocks for v in block]
    if any(not block for block in blocks) or sorted(flat) != list(range(hrep.n_vertices)):
        raise ValueError("budget blocks must partition all graph vertices into nonempty blocks")
    edges = set(hrep.edges)
    for block in blocks:
        for i, u in enumerate(block):
            for v in block[i + 1:]:
                if (min(u, v), max(u, v)) not in edges:
                    raise ValueError("each budget block must be a graph clique")
    rows = list(hrep.rows)
    for block in blocks:
        for sign in (1, -1):
            coeff = tuple(sign if v in block else 0 for v in range(hrep.n_vertices))
            proof = Derivation("budget_equation", context=(block, sign)) \
                if hrep.audit["provenance_recorded"] else None
            rows.append(Row(coeff, sign, proof))
    rows = tuple(sorted(set(rows), key=lambda r: (r.coeff, r.rhs)))
    audit = {**hrep.audit, "final_rows": len(rows), "budget_face": True,
             "budget_blocks": [list(block) for block in blocks],
             "capacity_domain": "complete independent transversals, via budget face"}
    return HRepresentation(hrep.n_vertices, hrep.edges, rows, audit, blocks)


def warp_hrep_from_signs(patch_budgets: Sequence[int], affordable: Sequence[Sequence[bool]], *,
                         limits: Limits | None = None, provenance: bool = True) -> HRepresentation:
    """Price-sign-only producer. affordable[e][j] means patch e lies below budget j.

    The caller supplies the full list of nonempty off-intersection patches.
    Own-budget affordability must be False because it is equality, not strict.
    Geometric realization/completeness of the supplied patch list is not inferred.
    """
    budgets = tuple(patch_budgets)
    if not budgets or any(type(j) is not int or j < 0 for j in budgets):
        raise ValueError("nonempty patch list and nonnegative integer budgets are required")
    count = max(budgets) + 1
    if set(budgets) != set(range(count)):
        raise ValueError("budget indices must be contiguous and have patches")
    signs = tuple(tuple(row) for row in affordable)
    if len(signs) != len(budgets) or any(len(row) != count for row in signs):
        raise ValueError("affordability array has incorrect dimensions")
    if any(type(sign) is not bool for row in signs for sign in row):
        raise TypeError("strict affordability entries must be booleans")
    if any(signs[e][j] for e, j in enumerate(budgets)):
        raise ValueError("own-budget entries represent equality, not strict affordability")
    edges = []
    for e in range(len(budgets)):
        for f in range(e + 1, len(budgets)):
            if budgets[e] == budgets[f] or (signs[e][budgets[f]] and signs[f][budgets[e]]):
                edges.append((e, f))
    base = stable_set_hrep(len(budgets), edges, limits=limits, provenance=provenance)
    return complete_budget_face(base, [[e for e, k in enumerate(budgets) if k == j]
                                      for j in range(count)])


def verify_arithmetic_provenance(hrep: HRepresentation) -> dict:
    """Check each retained row's exact arithmetic DAG, without numerical solvers.

    This certifies the documented row transformations, not a new proof of the
    semantic graph disjunction or the FM reconstruction/existential direction.
    """
    if not __debug__:
        raise RuntimeError("the arithmetic checker requires assertions enabled; no audit was performed")
    seen = set()
    pending = list(hrep.rows)
    counts = {}
    while pending:
        row = pending.pop()
        identity = id(row)
        if identity in seen:
            continue
        seen.add(identity)
        proof = row.proof
        if proof is None:
            raise ValueError("arithmetic provenance was not recorded")
        parents = proof.parents
        pending.extend(parents)
        counts[proof.rule] = counts.get(proof.rule, 0) + 1
        coeff, rhs = row.coeff, row.rhs
        if proof.rule == "box":
            vertex, sign = proof.context
            assert coeff == tuple(sign if j == vertex else 0 for j in range(len(coeff)))
            assert rhs == (1 if sign == 1 else 0)
        elif proof.rule == "budget_equation":
            block, sign = proof.context
            assert block in hrep.budget_blocks
            assert coeff == tuple(sign if j in block else 0 for j in range(len(coeff)))
            assert rhs == sign
        elif proof.rule == "normalize":
            parent, = parents
            assert proof.divisor > 0
            assert tuple(a * proof.divisor for a in coeff) == parent.coeff
            assert rhs * proof.divisor == parent.rhs
        elif proof.rule == "drop_zero":
            parent, = parents
            column, = proof.context
            assert parent.coeff[column] == 0
            assert coeff == parent.coeff[:column] + parent.coeff[column + 1:]
            assert rhs == parent.rhs
        elif proof.rule == "fm_pair":
            positive, negative = parents
            column, = proof.context
            a, b = proof.multipliers
            assert a > 0 and b > 0
            assert a == -negative.coeff[column] and b == positive.coeff[column]
            combined = tuple(a * p + b * n for p, n in zip(positive.coeff, negative.coeff))
            assert combined[column] == 0
            assert coeff == combined[:column] + combined[column + 1:]
            assert rhs == a * positive.rhs + b * negative.rhs
        elif proof.rule == "positive_sum":
            assert len(parents) == len(proof.multipliers)
            assert all(multiplier > 0 for multiplier in proof.multipliers)
            assert coeff == tuple(sum(weight * parent.coeff[j]
                                      for parent, weight in zip(parents, proof.multipliers))
                                  for j in range(len(coeff)))
            assert rhs == sum(weight * parent.rhs for parent, weight in zip(parents, proof.multipliers))
        elif proof.rule in ("omitted_perspective", "universal_lift"):
            parent, = parents
            vertex, remaining = proof.context
            expected = list(parent.coeff) + [-parent.coeff[v] for v in remaining]
            assert expected[vertex] == 0
            expected[vertex] = parent.rhs
            assert coeff == tuple(expected) and rhs == parent.rhs
            if proof.rule == "universal_lift":
                assert not remaining
        elif proof.rule == "selected_perspective":
            parent, = parents
            vertex, remaining = proof.context
            expected = [0] * len(parent.coeff) + [parent.coeff[v] for v in remaining]
            expected[vertex] = -parent.rhs
            assert coeff == tuple(expected) and rhs == 0
        else:
            raise ValueError(f"unknown arithmetic rule: {proof.rule}")
    for certificate in hrep.audit["redundancy_certificates"]:
        coeff, rhs = certificate["coeff"], certificate["rhs"]
        bounded = set(certificate["bounded_columns"])
        assert all(j in bounded for j, a in enumerate(coeff) if a)
        if certificate["rule"] == "unit_box_tautology":
            assert sum(max(a, 0) for a in coeff) <= rhs
        elif certificate["rule"] == "nonnegative_coordinate_dominance":
            assert certificate["stronger_rhs"] <= rhs
            assert all(a >= b for a, b in zip(certificate["stronger_coeff"], coeff))
        else:
            raise ValueError("unknown redundancy rule")
    return {"arithmetic_audit_passed": True, "dag_rows_checked": len(seen),
            "redundancy_certificates_checked": len(hrep.audit["redundancy_certificates"]),
            "rules": counts, "semantic_completeness": "separate graph/FM induction obligation"}
