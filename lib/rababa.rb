# frozen_string_literal: true

require "rababa/version"
require "rababa/arabic"
require "rababa/hebrew"
require "optparse"
require "onnxruntime"
require "yaml"
require "tqdm"

module Rababa
  class Error < StandardError; end
end
