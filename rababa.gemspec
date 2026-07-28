# frozen_string_literal: true

require_relative "lib/rababa/version"

Gem::Specification.new do |spec|
  spec.name = "rababa"
  spec.version = Rababa::VERSION
  spec.authors = ["Ribose"]
  spec.email = ["open.source@ribose.com"]

  spec.summary = "Middle Eastern Languages diacriticizer from Interscript."
  spec.description = "Middle Eastern Languages diacriticizer from Interscript."
  spec.homepage = "https://www.interscript.org"
  spec.required_ruby_version = ">= 3.3.0"

  spec.metadata["homepage_uri"]         = spec.homepage
  spec.metadata["source_code_uri"]      = "https://github.com/interscript/rababa"
  spec.metadata["changelog_uri"]        = "https://github.com/interscript/rababa/releases"
  spec.metadata["bug_tracker_uri"]      = "https://github.com/interscript/rababa/issues"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir.chdir(__dir__) do
    Dir[
      "lib/**/*",
      "exe/**/*",
      "config/**/*",
      "data/**/*",
      "models-data/**/*",
      "README*",
      "LICENSE*",
      "*.gemspec"
    ].select { |f| File.file?(f) }
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "onnxruntime"
  spec.add_dependency "optparse"
  spec.add_dependency "yaml"
  spec.add_dependency "tqdm"
end
