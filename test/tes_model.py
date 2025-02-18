import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
from code_ai import model

if __name__ == '__main__':
    model.Base.metadata.create_all(model.engine)
