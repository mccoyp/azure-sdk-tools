# ts-config-no-experimentaldecorators

Requires `compilerOptions.experimentalDecorators` in `tsconfig.json` to be set to `false`.

This rule is fixable using the `--fix` option.

## Examples

### Good

```json
{
  "compilerOptions": {
    "experimentalDecorators": false
  }
}
```

### Bad

```json
{
  "compilerOptions": {
    "experimentalDecorators": true
  }
}
```

```json
{
  "compilerOptions": {}
}
```

```json
{}
```

## [Source](https://azure.github.io/azure-sdk/typescript_design.html#ts-config-no-experimentaldecorators)
