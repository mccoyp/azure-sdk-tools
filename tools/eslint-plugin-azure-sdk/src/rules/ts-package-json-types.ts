/**
 * @file Rule to force the inclusion of type declarations in the package.
 * @author Arpan Laha
 */

import { Rule } from "eslint";
import { Literal, Property } from "estree";
import { getRuleMetaData, getVerifiers, stripPath } from "../utils";

//------------------------------------------------------------------------------
// Rule Definition
//------------------------------------------------------------------------------

export = {
  meta: getRuleMetaData(
    "ts-package-json-types",
    "force the inclusion of type declarations in the package"
  ),
  create: (context: Rule.RuleContext): Rule.RuleListener => {
    const verifiers = getVerifiers(context, {
      outer: "types",
      expected: false
    });
    return stripPath(context.getFilename()) === "package.json"
      ? ({
          // callback functions

          // check to see if types exists at the outermost level
          "ExpressionStatement > ObjectExpression": verifiers.existsInFile,

          // check the node corresponding to types to see if its value is a TypeScript declaration file
          "ExpressionStatement > ObjectExpression > Property[key.value='types']": (
            node: Property
          ): void => {
            if (node.value.type !== "Literal") {
              context.report({
                node: node.value,
                message: "types is not set to a string"
              });
            }
            const nodeValue = node.value as Literal;
            // filename ending in '.d.ts'
            if (!/\.d\.ts$/.test(nodeValue.value as string)) {
              context.report({
                node: nodeValue,
                message:
                  "provided types path is not a TypeScript declaration file"
              });
            }
          }
        } as Rule.RuleListener)
      : {};
  }
};
