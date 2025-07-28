import ast
import astunparse
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}
        self.classes = {}
        self.calls = defaultdict(set)
        self.defined_names = set()
        self.used_names = set()

    def visit_FunctionDef(self, node):
        self.functions[node.name] = node
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes[node.name] = node
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls[self.current_function_or_class].add(node.func.id)
            self.used_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.used_names.add(node.func.attr)
        self.generic_visit(node)

    def analyze(self, code):
        tree = ast.parse(code)
        self.current_function_or_class = "main"
        self.visit(tree)
        return tree

    def visit(self, node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            self.current_function_or_class = node.name
        super().visit(node)

    def detect_duplicates(self):
        func_dupes = {k: v for k, v in self.functions.items() if list(self.functions.keys()).count(k) > 1}
        class_dupes = {k: v for k, v in self.classes.items() if list(self.classes.keys()).count(k) > 1}
        return func_dupes, class_dupes

    def detect_unused(self):
        unused = self.defined_names - self.used_names
        return unused

    def build_dependency_graph(self):
        G = nx.DiGraph()
        for caller, callees in self.calls.items():
            for callee in callees:
                G.add_edge(caller, callee)
        return G

def visualize_graph(graph, title='依存関係グラフ'):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.savefig("dependency_graph.png")
    print("🖼️ 依存関係グラフを dependency_graph.png に保存しました。")
    plt.close()

def main(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] ファイルが見つかりません: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    analyzer = CodeAnalyzer()
    analyzer.analyze(code)

    # 重複チェック
    func_dupes, class_dupes = analyzer.detect_duplicates()
    if func_dupes or class_dupes:
        print("⚠️ 重複定義検出:")
        for name in set(func_dupes):
            print(f"  - 関数重複: {name}")
        for name in set(class_dupes):
            print(f"  - クラス重複: {name}")
    else:
        print("✅ 重複定義なし")

    # 未使用検出
    unused_defs = analyzer.detect_unused()
    if unused_defs:
        print("🧹 未使用関数/クラス:")
        for name in unused_defs:
            print(f"  - {name}")
    else:
        print("✅ すべての関数・クラスは使用されています")

    # 依存関係の可視化
    graph = analyzer.build_dependency_graph()
    visualize_graph(graph)

if __name__ == "__main__":
    # 解析したいファイルのパスを指定
    main("test2.py")  # ← 実際のスクリプトファイル名に変更
