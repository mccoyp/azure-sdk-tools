# eslint-plugin-azure-sdk

An ESLint plugin enforcing [design guidelines for the JavaScript/TypeScript Azure SDK](https://azure.github.io/azure-sdk/typescript_introduction.html).

## Installation

You'll first need to install [ESLint](http://eslint.org):

```shell
npm i eslint --save-dev
```

Next, install `@azure/eslint-plugin-azure-sdk`:

```shell
npm install @azure/eslint-plugin-azure-sdk --save-dev
```

## Usage

Add `@azure/azure-sdk` to the `plugins` section of your `.eslintrc` configuration file. You can omit the `eslint-plugin-` prefix:

```json
{
  "plugins": ["@azure/azure-sdk"]
}
```

Make sure to set your `.eslintrc` configuration file's `parserOptions.project` field to point to the tsconfig file at the root of your project as follows:

```json
{
  "parserOptions": {
    "project": "./tsconfig.json"
  }
}
```

For all rules to be enforced according to the standards set by the Design Guidelines, add this plugin's `recommended` configuration to the `extends` section of your `.eslintrc` configuration file as follows:

```json
{
  "extends": ["plugin:@azure/azure-sdk/recommended"]
}
```

If the main TypeScript entrypoint to your package is not in `src/index.ts`, set `settings.main` in your `.eslintrc` configuration file to the entrypoint as follows (for example, if the entrypoint is `index.ts`):

```json
{
  "settings": {
    "main": "index.ts"
  }
}
```

If you need to modify or disable specific rules, you can do so in the `rules` section of your `.eslintrc` configuration file. For example, if you are not targeting Node, disable `ts-config-moduleresolution` as follows:

```json
{
  "rules": {
    "@azure/azure-sdk/ts-config-moduleresolution": "off"
  }
}
```

Some rules (see table below) are fixable using the `--fix` ESLint option (added in `1.3.0`).

## Supported Rules

### Key

| Symbol                    | Meaning                     |
| ------------------------- | --------------------------- |
| :triangular_flag_on_post: | Error                       |
| :warning:                 | Warning                     |
| :heavy_multiplication_x:  | Off                         |
| :heavy_check_mark:        | Fixable and autofix-enabled |
| :x:                       | Not fixable                 |

### Rules

| Rule                                                                                                                                      | Default                   | Fixable            | Release |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------ | ------- |
| [**github-source-headers**](/tools/eslint-plugin-azure-sdk/docs/rules/github-source-headers.md)                                           | :triangular_flag_on_post: | :heavy_check_mark: | `1.1.0` |
| [**ts-apisurface-standardized-verbs**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-apisurface-standardized-verbs.md)                     | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-apisurface-supportcancellation**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-apisurface-supportcancellation.md)                   | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-config-allowsyntheticdefaultimports**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-allowsyntheticdefaultimports.md)         | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-declaration**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-declaration.md)                                           | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-esmoduleinterop**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-esmoduleinterop.md)                                   | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-exclude**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-exclude.md)                                                   | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-forceconsistentcasinginfilenames**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-forceconsistentcasinginfilenames.md) | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-importhelpers**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-importhelpers.md)                                       | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-lib**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-lib.md)                                                           | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-module**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-module.md)                                                     | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-moduleresolution**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-moduleresolution.md)                                 | :triangular_flag_on_post: | :heavy_check_mark: | `1.1.0` |
| [**ts-config-no-experimentaldecorators**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-no-experimentaldecorators.md)               | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-sourcemap**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-sourcemap.md)                                               | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-strict**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-strict.md)                                                     | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-config-target**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-config-target.md)                                                     | :triangular_flag_on_post: | :x:                | `1.1.0` |
| [**ts-doc-internal**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-doc-internal.md)                                                       | :triangular_flag_on_post: | :x:                | `1.1.0` |
| [**ts-error-handling**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-error-handling.md)                                                   | :heavy_multiplication_x:  | :x:                | `1.1.0` |
| [**ts-modules-only-named**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-modules-only-named.md)                                           | :triangular_flag_on_post: | :x:                | `1.1.0` |
| [**ts-naming-drop-noun**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-naming-drop-noun.md)                                               | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-naming-options**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-naming-options.md)                                                   | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-naming-subclients**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-naming-subclients.md)                                             | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-no-const-enums**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-no-const-enums.md)                                                   | :warning:                 | :heavy_check_mark: | `1.1.0` |
| [**ts-no-namespaces**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-no-namespaces.md)                                                     | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-package-json-author**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-author.md)                                         | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-package-json-bugs**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-bugs.md)                                             | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-package-json-engine-is-present**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-engine-is-present.md)                   | :triangular_flag_on_post: | :heavy_check_mark: | `1.1.0` |
| [**ts-package-json-files-required**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-files-required.md)                         | :triangular_flag_on_post: | :heavy_check_mark: | `1.1.0` |
| [**ts-package-json-homepage**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-homepage.md)                                     | :triangular_flag_on_post: | :x:                | `1.0.0` |
| [**ts-package-json-keywords**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-keywords.md)                                     | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-package-json-license**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-license.md)                                       | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-package-json-main-is-cjs**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-main-is-cjs.md)                               | :triangular_flag_on_post: | :heavy_check_mark: | `1.1.0` |
| [**ts-package-json-module**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-module.md)                                         | :triangular_flag_on_post: | :heavy_check_mark: | `1.1.0` |
| [**ts-package-json-name**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-name.md)                                             | :triangular_flag_on_post: | :x:                | `1.0.0` |
| [**ts-package-json-repo**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-repo.md)                                             | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-package-json-required-scripts**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-required-scripts.md)                     | :triangular_flag_on_post: | :x:                | `1.0.0` |
| [**ts-package-json-sideeffects**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-sideeffects.md)                               | :triangular_flag_on_post: | :heavy_check_mark: | `1.0.0` |
| [**ts-package-json-types**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-package-json-types.md)                                           | :triangular_flag_on_post: | :x:                | `1.1.0` |
| [**ts-pagination-list**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-pagination-list.md)                                                 | :triangular_flag_on_post: | :x:                | `1.2.0` |
| [**ts-use-interface-parameters**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-use-interface-parameters.md)                               | :warning:                 | :x:                | `1.1.0` |
| [**ts-use-promises**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-use-promises.md)                                                       | :triangular_flag_on_post: | :x:                | `1.1.0` |
| [**ts-versioning-semver**](/tools/eslint-plugin-azure-sdk/docs/rules/ts-versioning-semver.md)                                             | :triangular_flag_on_post: | :x:                | `1.1.0` |
