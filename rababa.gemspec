# frozen_string_literal: true

require_relative "lib/rababa/version"

Gem::Specification.new do |spec|
  spec.name = "rababa"
  spec.version = Rababa::VERSION
  spec.authors = ["Ribose"]
  spec.email = ["open.source@ribose.com"]

  spec.summary = "Middle Eastern Languages diacriticizer from Interscript."
  # spec.description   = "TODO: Write a longer description or delete this line."
  spec.homepage = "https://www.interscript.org"
  spec.required_ruby_version = Gem::Requirement.new(">= 2.5.0")

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/interscript/rababa"
  spec.metadata["changelog_uri"] = "https://github.com/interscript/rababa"

  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{\A(?:test|spec|features)/}) }
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "onnxruntime"
  spec.add_dependency "optparse"
  spec.add_dependency "yaml"
  spec.add_dependency "tqdm"
end
