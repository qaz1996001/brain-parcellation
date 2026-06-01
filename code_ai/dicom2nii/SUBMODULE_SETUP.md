# dicom2nii-rs Submodule Setup Guide

## Step 1: Push to GitHub Repository

The dicom2nii-rs code has been split into a separate branch. You need to push it to GitHub:

```bash
# From the brain-parcellation directory
cd /home/user/brain-parcellation

# Push the split branch to GitHub
git push https://github.com/qaz1996001/dicom2nii.git dicom2nii-rs-branch:main --force

# Or if you have SSH set up:
git push git@github.com:qaz1996001/dicom2nii.git dicom2nii-rs-branch:main --force
```

## Step 2: Remove the Current Directory

After pushing to GitHub, remove the current embedded directory:

```bash
# Remove the current dicom2nii-rs directory
rm -rf code_ai/dicom2nii/dicom2nii-rs

# Clean up the split branch
git branch -D dicom2nii-rs-branch
```

## Step 3: Add as Submodule

Add the GitHub repository as a submodule:

```bash
# Add submodule
git submodule add https://github.com/qaz1996001/dicom2nii.git code_ai/dicom2nii/dicom2nii-rs

# Initialize and update
git submodule init
git submodule update
```

## Step 4: Commit the Submodule Change

```bash
git add .gitmodules code_ai/dicom2nii/dicom2nii-rs
git commit -m "refactor: use dicom2nii-rs as git submodule

Move Rust implementation to separate repository for better maintainability.
Repository: https://github.com/qaz1996001/dicom2nii"

git push
```

## Working with Submodules

### Clone with Submodules
```bash
git clone --recurse-submodules <repo-url>
```

### Update Submodule to Latest
```bash
cd code_ai/dicom2nii/dicom2nii-rs
git pull origin main
cd ../../..
git add code_ai/dicom2nii/dicom2nii-rs
git commit -m "chore: update dicom2nii-rs submodule"
```

### Initialize Submodules (existing clone)
```bash
git submodule init
git submodule update
```
