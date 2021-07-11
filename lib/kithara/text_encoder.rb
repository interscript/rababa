"""
corresponds to: 
https://github.com/almodhfer/Arabic_Diacritization/blob/master/util/text_encoders.py
"""

require_relative "arabic_constant"

include arabic_constant


module text_encoder
    
    class TextEncoder 
        pad = "P"
    
        attr_accessor 
        def initialize(input_chars, target_charts, #: List[str]
                       cleaner_fn, #: Optional[str] = None
                       reverse_input: bool = false,
                       reverse_target: bool = false) 
        
            #if cleaner_fn:
            #    self.cleaner_fn = getattr(text_cleaners, cleaner_fn)
            #else:
            #    self.cleaner_fn = None
        
            @input_symbols: List[str] = [TextEncoder.pad] + input_chars
            @target_symbols: List[str] = [TextEncoder.pad] + target_charts

            @input_symbol_to_id: Hash = Hash.new([(s, i) for i,s in enumerate(@input_symbols)])
      
            @input_id_to_symbol: Hash = Hash.new([(i, s) for i,s in enumerate(@input_symbols)])

            @target_symbol_to_id: Hash = Hash.new([(s, i) for i,s in enumerate(@target_symbols)])
 
            @target_id_to_symbol: Hash = Hash.new([(i, s) for i,s in enumerate(@target_symbols)])

            @reverse_input = reverse_input
            @reverse_target = reverse_target
            @input_pad_id = @input_symbol_to_id[self.pad]
            @target_pad_id = @target_symbol_to_id[self.pad]
            @start_symbol_id = nil

        end 
        
    end
        
            
    class BasicArabicEncoder(TextEncoder) 
    
        attr_accessor 
        def initialize(cleaner_fn="basic_cleaners",
                       reverse_input: bool = false,
                       reverse_target: bool = false)
        
            input_chars = "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث".chars()
            target_charts = arabic_constant::ALL_POSSIBLE_HARAQAT.keys()
        
            super().__init__(input_chars,target_charts,
                             cleaner_fn=cleaner_fn,
                             reverse_input=reverse_input,
                             reverse_target=reverse_target)
        end
    end

            
    class ArabicEncoderWithStartSymbol(TextEncoder)
        
        attr_accessor
        def ititialize(cleaner_fn="basic_cleaners",
                       reverse_input: bool = false,
                       reverse_target: bool = false)
            
            input_chars: List[str] = list("بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث")
            # the only difference from the basic encoder is adding the start symbol
            target_charts = arabic_constant::ALL_POSSIBLE_HARAQAT.keys() + ["s"]

            super().__init__(input_chars,
                             target_charts,
                             cleaner_fn=cleaner_fn,
                             reverse_input=reverse_input,
                             reverse_target=reverse_target)

            @start_symbol_id = @target_symbol_to_id["s"]  

        end
        
end
