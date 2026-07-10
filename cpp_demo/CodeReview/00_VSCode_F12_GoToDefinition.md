# VS Code F12 Go To Definition 설정

macOS에서 `F12` 키가 VS Code의 `Go to Definition`이 아니라 스피커/볼륨 조절로 동작할 때, VS Code 단축키 설정으로 해결한 내용을 기록한다.

## 설정 방법

1. VS Code에서 `Cmd + Shift + P`를 누른다.
2. `Preferences: Open Keyboard Shortcuts`를 검색해서 실행한다.
3. 검색창에 `Go to Definition`을 입력한다.
4. `Go to Definition` 항목의 연필 아이콘을 클릭한다.
5. 원하는 단축키로 `F12`를 입력해서 등록한다.

## Keyboard Shortcuts JSON 예시

`Preferences: Open Keyboard Shortcuts (JSON)`에서 직접 설정할 경우 아래 내용을 추가한다.

```json
{
  "key": "f12",
  "command": "editor.action.revealDefinition",
  "when": "editorHasDefinitionProvider && editorTextFocus && !isInEmbeddedEditor"
}
```

## 참고

macOS 시스템 설정에서 `F1, F2 등의 키를 표준 기능 키로 사용`을 켜면 `F12`가 볼륨 키가 아니라 일반 기능 키로 동작한다.

이 경우 볼륨 조절은 `Fn + F12`로 사용할 수 있다.
