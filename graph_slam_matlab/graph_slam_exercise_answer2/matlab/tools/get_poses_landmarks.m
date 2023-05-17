% extract the offset of the poses and the landmarks

function [poses, landmarks] = get_poses_landmarks(g)

poses = [];
landmarks = [];


%% OCTAVE
%%%for [value, key] = g.idLookup
%%%  dim = value.dimension;
%%%  offset = value.offset;
  
fields = fieldnames(g.idLookup);

for i = 1:numel(fields)
    dim = g.idLookup.(fields{i}).dimension;
    offset = g.idLookup.(fields{i}).offset;
        
  if (dim == 3)
    poses = [poses; offset];
  elseif (dim == 2)
    landmarks = [landmarks; offset];
  end
end

end
