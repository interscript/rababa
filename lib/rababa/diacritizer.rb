module Rababa
  # A generic diacritizer to be overloaded by the respective languages.
  class Diacritizer
    def initialize(onnx_model_path, config)
      # load inference model from model_path
      @onnx_session = OnnxRuntime::InferenceSession.new(onnx_model_path)

      # load config
      @config = config
      @max_length = @config["max_len"]
      @batch_size = @config["batch_size"]

      # instantiate encoder's class
      @encoder = get_text_encoder
    end

    # Uses diacritize_text in a batch mode - and string mode
    # if we are past the batch size
    def diacritize_file(path)
      texts = File.read(path).split("\n").map(&:strip)

      # process batches
      out_texts = []
      idx = 0
      while idx + @batch_size <= texts.length
        originals = texts[idx...idx + @batch_size]

        out_texts += diacritize_text(originals)

        idx += @batch_size
      end

      # process rest of data
      while idx < texts.length
        out_texts += [diacritize_text(texts[idx])]
        idx += 1
      end

      out_texts
    end

    # Interface to implement by respective diactricizers

    def diacritize_text(text)
      raise NotImplementedError
    end

    def remove_diacritics(text)
      raise NotImplementedError
    end

    def get_text_encoder
      raise NotImplementedError
    end
  end
end
