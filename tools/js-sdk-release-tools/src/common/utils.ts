import shell from 'shelljs';
import path, { join, posix } from 'path';
import fs from 'fs';
import { SDKType, RunMode } from './types.js';
import { logger } from '../utils/logger.js';
import { Project, ScriptTarget, SourceFile } from 'ts-morph';
import { readFile } from 'fs/promises';
import { parse } from 'yaml';
import { access, readdir, rm, mkdir } from 'node:fs/promises';
import { SpawnOptions, spawn } from 'child_process';
import * as compiler from '@typespec/compiler';
import { dump, load as yamlLoad } from 'js-yaml';
import { NpmViewParameters, tryCreateLastestStableNpmViewFromGithub } from './npmUtils.js';

// ./eng/common/scripts/TypeSpec-Project-Process.ps1 script forces to use emitter '@azure-tools/typespec-ts',
// so do NOT change the emitter
const emitterName = '@azure-tools/typespec-ts';

// 1 hour in milliseconds unit
export const defaultChildProcessTimeout = 60 * 60 * 1000;

// TODO: remove it after we generate and use options by ourselves
const messageToTspConfigSample =
    'Please refer to https://github.com/Azure/azure-rest-api-specs/blob/main/specification/contosowidgetmanager/Contoso.WidgetManager/tspconfig.yaml for the right schema.';

const errorKeywordsInLowercase = new Set<string>(['error', 'err_pnpm_no_matching_version']);

function removeLastNewline(line: string): string {
    return line.replace(/\n$/, '')
}

function replaceAll(original: string, from: string, to: string) {
    return original.split(from).join(to);
}

function printErrorDetails(
    output: { stdout: string; stderr: string; code: number | null } | undefined,
    printDetails: boolean = false,
    errorAsWarning: boolean = false
) {
    if (!output) return;
    const getErrorSummary = (content: string) =>
        content
            .split('\n')
            .filter((line) => {
                for (const keyword of errorKeywordsInLowercase) {
                    if (line.toLowerCase().includes(keyword)) return true;
                }
                return false;
            })
            .map((line) => `  ${line}\n`);
    let summary = [...getErrorSummary(output.stderr), ...getErrorSummary(output.stdout)];
    logError(errorAsWarning)(`Exit code: ${output.code}`);
    if (summary.length > 0) {
        logError(errorAsWarning)(`Summary:`);
        summary.forEach((line) => logError(errorAsWarning)(removeLastNewline(line)));
    }
    if (printDetails) {
        const stderr = removeLastNewline(output.stderr);
        const stdout = removeLastNewline(output.stdout);
        logError(errorAsWarning)(`Details:`);
        if (stderr) {
            logError(errorAsWarning)(`  stderr:`);
            stderr.split('\n').forEach((line) => logger.warn(`    ${line}`));
        }
        if (stdout) {
            logError(errorAsWarning)(`  stdout:`);
            stdout.split('\n').forEach((line) => logger.warn(`    ${line}`));
        }
    }
}

function getDistClassicClientParametersPath(packageRoot: string): string {
    return path.join(packageRoot, 'dist/esm/models/parameters.js');
}

export const runCommandOptions: SpawnOptions = { shell: true, stdio: ['pipe', 'pipe', 'pipe'] };

function logError(errorAsWarning: boolean) {
    return errorAsWarning ? logger.warn : logger.error;
}

export function getClassicClientParametersPath(packageRoot: string): string {
    return path.join(packageRoot, 'src', 'models', 'parameters.ts');
}

// NOTE: due to migration tool, the folder structure is changed,
//       and src folder is removed in new packages, 
//       so we need to check both src and dist folders
export function getSDKType(packageRoot: string): SDKType {
    const packageName = getNpmPackageName(packageRoot);
    if (packageName.startsWith('@azure-rest/')) {
        return SDKType.RestLevelClient;
    }

    const srcParaPath = getClassicClientParametersPath(packageRoot);
    const distParaPath = getDistClassicClientParametersPath(packageRoot);

    const srcParameterExist = shell.test('-e', srcParaPath);
    const distParameterExist = shell.test('-e', distParaPath);

    const type = srcParameterExist || distParameterExist ? SDKType.HighLevelClient : SDKType.ModularClient;
    logger.info(`SDK type '${type}' is detected in '${packageRoot}'.`);
    return type;
}

export function getNpmPackageName(packageRoot: string): string {
    const packageJsonPath = path.join(packageRoot, 'package.json');
    const packageJson = fs.readFileSync(packageJsonPath, { encoding: 'utf-8' });
    const packageName = JSON.parse(packageJson).name;
    return packageName;
}

export function getApiReviewBasePath(packageRoot: string): string {
    const sdkType = getSDKType(packageRoot);
    const npmPackageName = getNpmPackageName(packageRoot);
    let apiReviewPath: string;
    switch (sdkType) {
        case SDKType.ModularClient:
            const modularPackageName = npmPackageName.substring('@azure/'.length);
            const apiViewFileName = `${modularPackageName}`;
            apiReviewPath = path.join(packageRoot, 'review', apiViewFileName);
            break;
        case SDKType.HighLevelClient:
        case SDKType.RestLevelClient:
        default:
            // only one xxx.api.md
            const packageName = npmPackageName.split('/')[1];
            apiReviewPath = path.join(packageRoot, 'review', `${packageName}`);
    }
    return apiReviewPath;
}

export function getApiReviewPath(packageRoot: string): string {
    const NODE_API_MD_SUFFIX = '-node.api.md';
    const API_REVIEW_SUFFIX = '.api.md';
    const apiReviewPath = getApiReviewBasePath(packageRoot);

    // First check if node.api.md exists
    const nodePath = `${apiReviewPath}${NODE_API_MD_SUFFIX}`;
    if (fs.existsSync(nodePath)) {
        logger.info(`Using node API review file: ${nodePath}`);
        return nodePath;
    }

    // If node.api.md doesn't exist, return the standard .api.md path
    const standardPath = `${apiReviewPath}${API_REVIEW_SUFFIX}`;
    return standardPath;
}

export function getTsSourceFile(filePath: string): SourceFile | undefined {
    const target = ScriptTarget.ES2015;
    const compilerOptions = { target };
    const project = new Project({ compilerOptions });
    project.addSourceFileAtPath(filePath);
    return project.getSourceFile(filePath);
}

// changelog policy: https://aka.ms/azsdk/guideline/changelogs
export function fixChangelogFormat(content: string) {
    content = replaceAll(content, '**Features**', '### Features Added')!;
    content = replaceAll(content, '**Breaking Changes**', '### Breaking Changes')!;
    content = replaceAll(content, '**Bugs Fixed**', '### Bugs Fixed')!;
    content = replaceAll(content, '**Other Changes**', '### Other Changes')!;
    return content;
}

export function tryReadNpmPackageChangelog(changelogPathFromNpm: string, NpmViewParameters?: NpmViewParameters): string {
    try {
        if (!fs.existsSync(changelogPathFromNpm)) {
            logger.warn(`Failed to find NPM package's changelog '${changelogPathFromNpm}'`);
            if (NpmViewParameters) {
                tryCreateLastestStableNpmViewFromGithub(NpmViewParameters);
            }
            else {
                return ""
            }
        }
        const originalChangeLogContent = fs.readFileSync(changelogPathFromNpm, { encoding: 'utf-8' });
        return originalChangeLogContent;
    } catch (err) {
        logger.warn(`Failed to read NPM package's changelog '${changelogPathFromNpm}': ${(err as Error)?.stack ?? err}`);
        return '';
    }
}

export async function isMgmtPackage(typeSpecDirectory: string): Promise<Boolean> {
    const mainTspPath = join(typeSpecDirectory, 'main.tsp');
    const content = await readFile(mainTspPath, { encoding: 'utf-8' });
    if (!content) {
        throw new Error(`Failed to get main.tsp in ${typeSpecDirectory}`);
    }
    return content.includes("armProviderNamespace");
}

export async function loadTspConfig(typeSpecDirectory: string): Promise<Exclude<any, null | undefined>> {
    const configPath = join(typeSpecDirectory, 'tspconfig.yaml');
    const content = await readFile(configPath, { encoding: 'utf-8' });
    const config = parse(content.toString());
    if (!config) {
        throw new Error(`Failed to parse tspconfig.yaml in ${typeSpecDirectory}`);
    }
    return config;
}

// generated path is in posix format
// e.g. sdk/mongocluster/arm-mongocluster
export async function getGeneratedPackageDirectory(typeSpecDirectory: string, sdkRepoRoot: string): Promise<string> {
    const tspConfig = await resolveOptions(typeSpecDirectory);
    let packageDir = tspConfig.configFile.parameters?.["package-dir"]?.default;
    let serviceDir = tspConfig.configFile.parameters?.["service-dir"]?.default;
    const emitterOptions = tspConfig.options?.[emitterName];
    const serviceDirFromEmitter = emitterOptions?.['service-dir'];
    if (serviceDirFromEmitter) {
        serviceDir = serviceDirFromEmitter;
    }
    const packageDirFromEmitter = emitterOptions?.['package-dir'];
    if (packageDirFromEmitter) {
        packageDir = packageDirFromEmitter;
    }
    if (!serviceDir) {
        throw new Error(`Miss service-dir in parameters section of tspconfig.yaml. ${messageToTspConfigSample}`);
    }
    if (!packageDir) {
        throw new Error(`Miss package-dir in ${emitterName} options of tspconfig.yaml. ${messageToTspConfigSample}`);
    }
    const packageDirFromRoot = posix.join(sdkRepoRoot, serviceDir, packageDir);
    return packageDirFromRoot;
}


export async function runCommand(
    command: string,
    args: readonly string[],
    options: SpawnOptions = runCommandOptions,
    realtimeOutput: boolean = true,
    timeoutSeconds: number | undefined = undefined,
    errorAsWarning: boolean = false
): Promise<{ stdout: string; stderr: string; code: number | null }> {
    let stdout = '';
    let stderr = '';
    const commandStr = `${command} ${args.join(' ')}`;
    logger.info(`Start to run command: '${commandStr}'.`);
    const child = spawn(command, args, options);

    let timedOut = false;
    const timer =
        timeoutSeconds &&
        setTimeout(() => {
            timedOut = true;
            child.kill();
            throw new Error(`Process timed out after ${timeoutSeconds}s`);
        }, timeoutSeconds * 1000);

    child.stdout?.setEncoding('utf8');
    child.stderr?.setEncoding('utf8');

    child.stdout?.on('data', (data) => {
        const str = data.toString();
        stdout += str;
        if (realtimeOutput) logger.info(str);
    });

    child.stderr?.on('data', (data) => {
        const str = data.toString();
        stderr += str;
        if (realtimeOutput) logger.warn(str);
    });

    let resolve: (value: void | PromiseLike<void>) => void;
    let reject: (reason?: any) => void;
    const promise = new Promise<void>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    let code: number | null = 0;

    child.on('exit', (exitCode, signal) => {
        if (timer) clearTimeout(timer);
        if (timedOut || !signal) { return; }
        logError(errorAsWarning)(`Command '${commandStr}' exited with signal '${signal ?? 'SIGTERM'}' and code ${exitCode}.`);
    });

    child.on('close', (exitCode) => {
        if (exitCode === 0) {
            resolve();
            logger.info(`Command '${commandStr}' closed with code '${exitCode}'.`);
            return;
        }
        code = exitCode;
        logError(errorAsWarning)(`Command closed with code '${exitCode}'.`);
        printErrorDetails({ stdout, stderr, code: exitCode }, !realtimeOutput, errorAsWarning);
        reject(Error(`Command closed with code '${exitCode}'.`));

    });

    child.on('error', (err) => {
        logError(errorAsWarning)((err as Error)?.stack ?? err);
        printErrorDetails({ stdout, stderr, code: null }, !realtimeOutput, errorAsWarning);
        reject(err);
    });

    await promise;
    return { stdout, stderr, code };
}

export async function existsAsync(path: string): Promise<boolean> {
    try {
        await access(path);
        return true;
    } catch (error) {
        logger.warn(`Fail to find ${path} for error: ${error}`);
        return false;
    }
}

export async function resolveOptions(typeSpecDirectory: string): Promise<Exclude<any, null | undefined>> {
    const [{ config, ...options }, diagnostics] = await compiler.resolveCompilerOptions(
        compiler.NodeHost,
        {
            cwd: process.cwd(),
            entrypoint: typeSpecDirectory, // not really used here
            configPath: typeSpecDirectory,
        });
    return options
}

export function specifyApiVersionToGenerateSDKByTypeSpec(typeSpecDirectory: string, apiVersion: string) {
    const tspConfigPath = path.join(typeSpecDirectory, 'tspconfig.yaml');
    if (!fs.existsSync(tspConfigPath)) {
        throw new Error(`Failed to find tspconfig.yaml in ${typeSpecDirectory}.`);
    }

    const tspConfigContent = fs.readFileSync(tspConfigPath, 'utf8');

    let tspConfig;
    try {
        tspConfig = yamlLoad(tspConfigContent);
    } catch (error) {
        throw new Error(`Failed to parse tspconfig.yaml: ${error}`);
    }    

    const emitterOptions = tspConfig.options?.[emitterName];
    if (!emitterOptions) {
        throw new Error(`Failed to find ${emitterName} options in tspconfig.yaml.`);
    }

    const apiVersionInTspConfig = emitterOptions['api-version'];
    if (apiVersionInTspConfig !== apiVersion) {
        logger.warn(`The specified api-version ${apiVersion} is going to override ${apiVersionInTspConfig} in tspconfig.yaml`);
        emitterOptions['api-version'] = apiVersion;
        const updatedTspConfigContent = dump(tspConfig);
        fs.writeFileSync(tspConfigPath, updatedTspConfigContent, 'utf8');
    }

    logger.info(`Use api-version: ${apiVersion} to generate SDK.`);
}

// Get the spec repo where the project is defined to set into tsp-location.yaml
export function generateRepoDataInTspLocation(repoUrl: string) {
    return repoUrl.replace("https://github.com/", "")
}

/**
 * Removes entries from a directory, with optional filtering of entries to preserve
 * @param directory - Directory to clean up
 * @param entriesToPreserve - Optional array of entry names to preserve (not remove)
 * @returns Promise that resolves when cleanup is complete
 */
export async function cleanUpDirectory(
    directory: string, 
    entriesToPreserve: string[] = []
): Promise<void> {      
    // Check if directory exists first
    if (!fs.existsSync(directory)) {
        logger.info(`Directory ${directory} doesn't exist, nothing to clean up.`);
        return;
    }
    
    // If nothing to preserve, remove the entire directory and create an empty one
    if (entriesToPreserve.length === 0) {
        logger.info(`Completely cleaning ${directory} directory and recreating it empty`);
        await rm(directory, { recursive: true, force: true });
        await mkdir(directory, { recursive: true });
        return;
    }
    
    // If we need to preserve some entries, selectively remove others
    logger.info(`Cleaning ${directory} directory, preserving: ${entriesToPreserve.join(', ')}`);
    
    // Get all subdirectories and files
    const entries = await readdir(directory);
    
    // Filter entries to exclude those that should be preserved
    const filteredEntries = entries.filter(entry => !entriesToPreserve.includes(entry));

    // Process each entry
    for (const entry of filteredEntries) {
        const entryPath = posix.join(directory, entry);
        await rm(entryPath, { recursive: true, force: true });
    }
}

/**
 * Cleans up a package directory based on the run mode
 * @param packageDirectory - Package directory to clean up
 * @param runMode - Current run mode determining what to preserve
 * @returns Promise that resolves when cleanup is complete
 */
export async function cleanUpPackageDirectory(
    packageDirectory: string,
    runMode: RunMode,
): Promise<void> {
    // Preserve test directory and assets.json file only in Release and Local modes
    // In SpecPullRequest and Batch modes, remove everything
    const shouldPreserveTestAndAssets = runMode !== RunMode.SpecPullRequest && runMode !== RunMode.Batch;
    const entriesToPreserve = shouldPreserveTestAndAssets ? ["test", "assets.json"] : [];

    await cleanUpDirectory(packageDirectory, entriesToPreserve);
}

export async function getPackageNameFromTspConfig(typeSpecDirectory: string): Promise<string | undefined> {
    const tspConfig = await resolveOptions(typeSpecDirectory);
    const emitterOptions = tspConfig.options?.[emitterName];
    
    // Get from package-details.name which is the actual NPM package name
    if (emitterOptions?.['package-details']?.name) {
        return emitterOptions['package-details'].name;
    }
    
    return undefined;
}