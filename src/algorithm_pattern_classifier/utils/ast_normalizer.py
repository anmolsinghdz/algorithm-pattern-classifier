import ast


class ASTNormalizer(ast.NodeTransformer):
    """Pre-processes and normalizes Python AST structures before classification.

    This simplifies complex syntaxes (e.g. tuple unpacking assignments) to make
    pattern detection more robust.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
        """Normalize assignments by desugaring tuple unpackings if possible.

        Args:
            node: The Assign node to normalize.

        Returns:
            The normalized AST node or list of nodes.
        """
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Tuple)
            and len(node.targets[0].elts) == len(node.value.elts)
        ):
            new_nodes: list[ast.AST] = []
            for target, val in zip(node.targets[0].elts, node.value.elts, strict=True):
                new_assign = ast.Assign(targets=[target], value=val)
                ast.copy_location(new_assign, node)
                new_nodes.append(new_assign)

            # Recursively visit the newly created nodes
            resolved_nodes: list[ast.AST] = []
            for n in new_nodes:
                visited = self.visit(n)
                if isinstance(visited, list):
                    resolved_nodes.extend(visited)
                elif visited is not None:
                    resolved_nodes.append(visited)
            return resolved_nodes

        return self.generic_visit(node)
