import argparse, ast, json, os, random, keyword, builtins, textwrap

BUILTINS = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls"}

class _ScopeLocals(ast.NodeVisitor):

    def __init__(self):
        self.bound = set(); self.skip = set()
    def visit_FunctionDef(self, node): pass
    def visit_AsyncFunctionDef(self, node): pass
    def visit_Lambda(self, node): pass
    def visit_ClassDef(self, node): pass
    def visit_Global(self, node): self.skip.update(node.names)
    def visit_Nonlocal(self, node): self.skip.update(node.names)
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store): self.bound.add(node.id)
    def visit_arg(self, node): self.bound.add(node.arg)

class _Renamer(ast.NodeTransformer):

    def __init__(self, mapping):
        self.m = mapping; self._depth = 0
    def _opaque(self, node):
        self._depth += 1; self.generic_visit(node); self._depth -= 1; return node
    def visit_FunctionDef(self, node):
        if self._depth == 0:
            for a in list(node.args.args) + list(node.args.kwonlyargs) + list(node.args.posonlyargs):
                if a.arg in self.m: a.arg = self.m[a.arg]
            if node.args.vararg and node.args.vararg.arg in self.m:
                node.args.vararg.arg = self.m[node.args.vararg.arg]
            if node.args.kwarg and node.args.kwarg.arg in self.m:
                node.args.kwarg.arg = self.m[node.args.kwarg.arg]
            return self._opaque(node)
        return node
    def visit_AsyncFunctionDef(self, node): return self.visit_FunctionDef(node)
    def visit_Lambda(self, node): return node
    def visit_ClassDef(self, node): return node
    def visit_Name(self, node):
        if node.id in self.m: node.id = self.m[node.id]
        return node

def _fresh(existing, i):
    n = f"aug_v{i}"
    while n in existing: i += 1; n = f"aug_v{i}"
    return n

def rename_locals(tree, rng, ratio=0.7):

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        sl = _ScopeLocals(); sl.generic_visit(fn)
        cands = sorted((sl.bound - sl.skip - BUILTINS))
        cands = [c for c in cands if not c.startswith("__")]
        if not cands: continue
        k = max(1, int(len(cands) * ratio))
        chosen = rng.sample(cands, min(k, len(cands)))
        mapping = {}
        for idx, name in enumerate(chosen):
            nn = _fresh(used, len(used) + idx); mapping[name] = nn; used.add(nn)
        if mapping:
            _Renamer(mapping).visit(fn)
    return tree

def insert_deadcode(tree, rng):

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        nm = _fresh(used, rng.randint(100, 999)); used.add(nm)
        stmt = ast.parse(f"{nm} = None").body[0]
        fn.body.insert(0, stmt)
    return tree

class _DeMorgan(ast.NodeTransformer):

    def __init__(self, rng, prob=0.8): self.rng = rng; self.prob = prob
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, ast.Not) and isinstance(node.operand, ast.BoolOp)
                and self.rng.random() < self.prob):
            bo = node.operand
            newop = ast.Or() if isinstance(bo.op, ast.And) else ast.And()
            return ast.BoolOp(op=newop, values=[ast.UnaryOp(op=ast.Not(), operand=v) for v in bo.values])
        return node

class _TempVar(ast.NodeTransformer):

    def __init__(self, rng, used, prob=0.4): self.rng = rng; self.used = used; self.i = 0; self.prob = prob
    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and not isinstance(node.value, (ast.Name, ast.Constant)) and self.rng.random() < self.prob):
            t = _fresh(self.used, 500 + self.i); self.i += 1; self.used.add(t)
            tmp = ast.Assign(targets=[ast.Name(id=t, ctx=ast.Store())], value=node.value)
            node.value = ast.Name(id=t, ctx=ast.Load())
            return [tmp, node]
        return node

def wrap_control(tree, rng, prob=0.5):

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if fn.body and rng.random() < prob:
            fn.body = [ast.If(test=ast.Constant(value=True), body=fn.body, orelse=[])]
    return tree

def augment_once(code, rng, enhanced=False):
    tree = ast.parse(code)
    if rng.random() < 0.9:
        tree = rename_locals(tree, rng, ratio=rng.choice([0.5, 0.7, 1.0]))
    if enhanced and rng.random() < 0.7:
        tree = _DeMorgan(rng).visit(tree)
    if enhanced and rng.random() < 0.6:
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        tree = _TempVar(rng, used).visit(tree)
    if rng.random() < 0.5:
        tree = insert_deadcode(tree, rng)
    if enhanced and rng.random() < 0.5:
        tree = wrap_control(tree, rng)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    ast.parse(out)
    return out

def _parse(code): return ast.parse(textwrap.dedent(code))
def _emit(tree):
    ast.fix_missing_locations(tree); out = ast.unparse(tree); ast.parse(out); return out
def _t_tempvar(code, rng):
    tree = _parse(code); used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return _emit(_TempVar(rng, used, prob=1.0).visit(tree))

PER_TRANSFORMS = {
    'rename':   lambda code, rng: _emit(rename_locals(_parse(code), rng, ratio=1.0)),
    'deadcode': lambda code, rng: _emit(insert_deadcode(_parse(code), rng)),
    'demorgan': lambda code, rng: _emit(_DeMorgan(rng, prob=1.0).visit(_parse(code))),
    'tempvar':  _t_tempvar,
    'wrap':     lambda code, rng: _emit(wrap_control(_parse(code), rng, prob=1.0)),
}
PER_ORDER = ['rename', 'deadcode', 'demorgan', 'tempvar', 'wrap']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=1, help='augmented variants per sample')
    ap.add_argument('--enhanced', action='store_true',
                    help='composed mode: stack extra transforms (De Morgan, temp-var, wrap-control) per variant')
    ap.add_argument('--per_transform', action='store_true',
                    help='expanded mode: emit ONE variant per transform (rename/deadcode/demorgan/tempvar/wrap), each in isolation')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = [json.loads(l) for l in open(a.inp)]
    n_in = len(rows); n_parse_fail = 0; n_emit = 0
    with open(a.out, 'w') as f:
        for i, r in enumerate(rows):
            code = textwrap.dedent(r.get('code', ''))
            try:
                ast.parse(code)
            except Exception:
                n_parse_fail += 1; continue
            if a.per_transform:
                seen = set()
                for op in PER_ORDER:
                    try:
                        v = PER_TRANSFORMS[op](code, rng)
                    except Exception:
                        continue
                    if v == code or v in seen:
                        continue
                    seen.add(v)
                    rec = dict(r); rec['code'] = v
                    rec['aug'] = True; rec['aug_of'] = i; rec['aug_op'] = op
                    f.write(json.dumps(rec) + '\n'); n_emit += 1
                continue
            seen = {code}
            tries = 0
            made = 0
            while made < a.k and tries < a.k * 4:
                tries += 1
                try:
                    aug = augment_once(code, rng, enhanced=a.enhanced)
                except Exception:
                    continue
                if aug in seen:
                    continue
                seen.add(aug)
                rec = dict(r); rec['code'] = aug
                rec['aug'] = True; rec['aug_of'] = i; rec['aug_variant'] = made
                f.write(json.dumps(rec) + '\n'); n_emit += 1; made += 1
    print(f"in={n_in} parse_fail={n_parse_fail} emitted={n_emit} -> {a.out}")

if __name__ == '__main__':
    main()
