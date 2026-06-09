import subprocess
import sys
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_vector_knowledge_base.py",
     "tests/test_hybrid_search.py",
     "tests/test_recommender.py",
     "-v", "--tb=short"],
    capture_output=True, text=True
)
with open("e:/vibecodeing/claude/test/all_tests_output.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nRETURN CODE: {result.returncode}\n")
