# this refers to:
#   https://github.com/interscript/rababa/blob/main/python/diacritizer.py
# as well a drastic simplification of
#   https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py

require "rababa/arabic/encoders"
require "rababa/arabic/reconciler"
require "rababa/diacritizer"

module Rababa
  module Arabic
    class Diacritizer < Rababa::Diacritizer
      include Arabic::Reconciler

      # preprocess text into indices
      def preprocess_text(text)
        # Warn user if text exceeds max_length.
        if text.length > @max_length
          text = text[0..@max_length]
          warn("WARNING:: string cut length > #{@max_length}:\n")
          warn("text:: " + text)
        end

        text = @encoder.clean(text)
        text = remove_diacritics(text)

        # correct expected length for vectors with 0's
        @encoder.input_to_sequence(text)
      end

      def remove_diacritics(text)
        text_dup = text.dup
        BASIC_HARAQAT.keys.each do |diacritic|
          text_dup.gsub!(diacritic, "")
        end

        text_dup
      end

      # Diacritize single arabic strings or a batch
      def diacritize_text(text)
        if text.is_a?(String)
          text = [text.strip] * @batch_size
          seq = [preprocess_text(text.first)] * @batch_size
          return_single = true
        else
          seq = text.map { |i| preprocess_text(i) }
        end

        # initialize onnx computation
        # redundancy caused by batch processing of nnets
        ort_inputs = {
          "src" => seq,
          "lengths" => seq.map(&:length)
        }

        # onnx predictions
        preds = predict_batch(ort_inputs)

        size = return_single ? 1 : @batch_size

        out = size.times.map do |i|
          reconcile_strings(
            text[i],
            combine_text_and_haraqat(seq[i], preds[i])
          )
        end

        return_single ? out.first : out
      end

      # Call ONNX model with data transformed in batches
      def predict_batch(batch_data)
        # onnx predictions
        predicts = @onnx_session.run(nil, batch_data)

        predicts[0].map do |p|
          p.map do |r|
            r.each_with_index.max[1]
          end
        end
      end

      # Combine: text + Haraqats --> diacritised arabic
      def combine_text_and_haraqat(vec_txt, vec_haraqat, encoding_mode = "std")
        if vec_txt.length != vec_haraqat.length
          raise "haraqat.len != txt.len in combine_text_and_haraqat"
        end

        text, i = "", 0
        loop do
          txt = vec_txt[i]
          haraq = vec_haraqat[i]
          i += 1
          break if (i == vec_txt.length) ||
            (txt == @encoder.input_pad_id)

          if encoding_mode == "std"
            s = @encoder.input_id_to_symbol[txt].to_s +
              @encoder.target_id_to_symbol[haraq].to_s

          # TODO: This following code is not used
          elsif encoding_mode == "escaped unicode"
            s = @encoder.input_id_to_symbol[txt].to_s +
              @utarget_symbol_to_id.utarget_id_to_symbol[haraq].to_s
          end
          text += s
        end

        text # .reverse
      end

      # Initialise text encoder from config params
      def get_text_encoder
        case @config["text_encoder"]
        when "BasicArabicEncoder"
          Arabic::Encoders::BasicArabicEncoder.new(@config["text_cleaner"])
        when "ArabicEncoderWithStartSymbol"
          Arabic::Encoders::ArabicEncoderWithStartSymbol.new(@config["text_cleaner"])
        else
          raise "text_encoder not found: #{@config["text_encoder"]}"
        end
      end
    end
  end
end
