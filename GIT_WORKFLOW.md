# 本地 → GitHub 推送流程

仓库已经连好远程(`origin` → `github.com/hanshuo-shuo/crash_bench.git`),
所以每次更新只需要三步:**看 → 存 → 推**。

---

## 每次更新代码后的标准流程

```bash
# 1. 看:改了哪些文件(? = 新文件,M = 改过的文件)
git status

# 2. 存:把改动加入并提交(commit = 本地的一次存档)
git add .                       # 把所有改动加进来(也可只加某个文件: git add PLAN.md)
git commit -m "写清楚这次改了啥"   # 例如 "Add implementation plan"

# 3. 推:上传到 GitHub
git push
```

就这三条。`git push` 之后刷新 GitHub 页面就能看到。

---

## 现在这次(推送 PLAN.md)具体命令

```bash
git add PLAN.md GIT_WORKFLOW.md
git commit -m "Add implementation plan and git workflow doc"
git push
```

---

## 常用辅助命令

```bash
git log --oneline -5      # 看最近 5 次提交历史
git diff                  # 看「还没 add」的改动具体内容
git diff --staged         # 看「已经 add、待 commit」的改动
git pull                  # 从 GitHub 拉最新(在别的机器上改过、或多人协作时先 pull)
git restore <文件>         # 放弃某个文件还没提交的改动(谨慎)
```

---

## 一些约定 / 注意点

- **commit message 写人话**:未来的你/合作者靠它快速看懂每次改了什么。
  例:`Add 5 pilot scenarios`、`Fix crash predicate threshold`。
- **多台机器**:换一台机器开工前先 `git pull`,避免冲突。
- **大文件别进 git**:模型权重、数据集、视频等大文件不要 commit。用 `.gitignore` 排除,
  必要时用 Git LFS 或网盘。建议尽早加一个 `.gitignore`(见下)。
- **第一次 push 报错要登录**:如果提示认证失败,用 GitHub 的 Personal Access Token
  当密码,或配置 SSH key(一次配好,以后免密)。

### 建议的 .gitignore(Python 项目)

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
*.pkl            # 如果 scenario 状态文件很小可以去掉这行
checkpoints/
data/
*.mp4
.DS_Store
```

---

## 万一遇到的情况

| 情况 | 怎么办 |
| --- | --- |
| `push` 被拒,提示 `rejected ... fetch first` | 别人/别的机器先推过了:先 `git pull` 再 `git push` |
| commit 写错了 message(还没 push) | `git commit --amend -m "新message"` |
| add 错了文件(还没 commit) | `git restore --staged <文件>` |
| 想撤销最后一次 commit 但保留改动 | `git reset --soft HEAD~1` |
