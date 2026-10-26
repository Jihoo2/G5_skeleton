# G5_skeleton

G5 팀 교과 실습 저장소입니다.

| 폴더 | 교과 |
|---|---|
| `subject9_machinevision/` | 교과 9 — OpenCV 기반 제조 머신비전 검사 (10일) |

## 브랜치 규칙

- `main` — 교과 폴더 뼈대만 관리합니다. **실습 코드는 main에 직접 올리지 않습니다.**
- 팀원 개인 브랜치 — 각자 `main`에서 브랜치를 만들어 자기 실습을 올립니다.

```bash
git checkout main
git pull
git checkout -b <본인브랜치>      # 예: seonghwan/subject9
# ... 작업 ...
git push -u origin <본인브랜치>
```

데이터(`data/`, `samples/`)와 실행 결과(`outputs/`)는 각 교과 폴더의 `.gitignore`로 제외됩니다.
