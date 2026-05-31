# python -m unittest tests.test_k_anonymity

# coverage run --source=anonymization_lib -m unittest discover -s tests -p "test_*.py" -t .
# coverage run --source=anonymization_lib -m unittest tests.test_generalization


# coverage report -m
# coverage html
# firefox htmlcov/index.html

# python -m build


# ================================
# VER TAGS
# ================================

# Listar todas las tags
# git tag

# Listar tags con mensaje
# git tag -n


# ================================
# CREAR TAGS
# ================================

# Crear una tag simple (no recomendado para releases)
# git tag v0.4.1

# Crear una tag anotada (RECOMENDADO)
# git tag -a v0.4.1 -m "Patch: improve tests and validations"

# Crear una tag en el último commit (HEAD)
# git tag -a v0.4.1 -m "Patch release"

# Crear una tag en un commit específico
# git tag -a v0.3.0 abc123 -m "Release substitution feature"


# ================================
# VER INFORMACIÓN DE UNA TAG
# ================================

# Mostrar detalles de una tag
# git show v0.4.1


# ================================
# SUBIR TAGS
# ================================

# Subir una tag concreta al remoto
# git push origin v0.4.1

# Subir todas las tags
# git push origin --tags


# ================================
# BORRAR TAGS
# ================================

# Borrar una tag local
# git tag -d v0.4.1

# Borrar una tag en remoto
# git push origin --delete v0.4.1


# ================================
# RECREAR UNA TAG (CASO TÍPICO)
# ================================

# 1. Borrar la tag local
# git tag -d v0.4.1

# 2. Borrar la tag en remoto
# git push origin --delete v0.4.1

# 3. Crear la nueva tag
# git tag -a v0.4.1 -m "Corrected release"

# 4. Subir la nueva tag
# git push origin v0.4.1


# ================================
# USAR TAGS
# ================================

# Cambiar a una tag (modo detached HEAD)
# git checkout v0.4.1

# Crear una rama a partir de una tag
# git checkout -b hotfix-from-v0.4.1 v0.4.1


# ================================
# ORDENAR TAGS
# ================================

# Listar tags ordenadas por versión (de mayor a menor)
# git tag --sort=-v:refname


# ================================
# FLUJO TÍPICO DE RELEASE
# ================================

# 1. Añadir cambios
# git add .

# 2. Commit
# git commit -m "Patch: improve tests and validations"

# 3. Crear tag
# git tag -a v0.4.1 -m "Patch release: tests + validation improvements"

# 4. Subir código
# git push origin main

# 5. Subir tag
# git push origin v0.4.1




