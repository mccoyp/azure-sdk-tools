// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Threading.Tasks;
using Xunit;

namespace Azure.ClientSdk.Analyzers.Tests
{
    public class AZC0001Tests
    {
        private readonly DiagnosticAnalyzerRunner _runner = new DiagnosticAnalyzerRunner(new ClientAssemblyNamespaceAnalyzer());

        [Fact]
        public async Task AZC0001ProducedForInvalidNamespaces()
        {
            var testSource = TestSource.Read(@"
namespace /*MM*/RandomNamespace
{
    public class Program { }
}
");
            var diagnostics = await _runner.GetDiagnosticsAsync(testSource.Source);
            var diagnostic = Assert.Single(diagnostics);
            Assert.Equal("AZC0001", diagnostic.Id);
            Assert.Equal("Namespace 'RandomNamespace' shouldn't contain public types. Use one of the following pre-approved namespace groups: " +
                         "Azure.ApplicationModel, Azure.Analytics, Azure.Data, Azure.Iot, Azure.Media, Azure.Messaging, Azure.ML, Azure.Security, Azure.Storage", diagnostic.GetMessage());

            AnalyzerAssert.DiagnosticLocation(testSource.DefaultMarkerLocation, diagnostic.Location);
        }

        [Fact]
        public async Task AZC0001ProducedOneErrorPerNamspaceDefinition()
        {
            var testSource = TestSource.Read(@"
namespace RandomNamespace
{
    public class Program { }
}

namespace RandomNamespace
{
    public class Program2 { }
}
");
            var diagnostics = await _runner.GetDiagnosticsAsync(testSource.Source);
            Assert.Equal(2, diagnostics.Length);
            Assert.All(diagnostics, d => Assert.Equal("AZC0001", d.Id));
        }

        [Fact]
        public async Task AZC0001NotProducedForNamespacesWithPrivateMembersOnly()
        {
            var testSource = TestSource.Read(@"
namespace RandomNamespace
{
    internal class Program { }
}
");
            var diagnostics = await _runner.GetDiagnosticsAsync(testSource.Source);
            Assert.Empty(diagnostics);
        }

        [Fact]
        public async Task AZC0001NotProducedForAllowedNamespaces()
        {
            var testSource = TestSource.Read(@"
namespace Azure.Storage.Hello
{
    public class Program { }
}
");
            var diagnostics = await _runner.GetDiagnosticsAsync(testSource.Source);
            Assert.Empty(diagnostics);
        }
    }
}