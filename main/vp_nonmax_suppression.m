function rect_scores = vp_nonmax_suppression(win_w, win_h, rect_count, rect, rect_scores)
    rect_scores = vp_nonmax_custom(win_w, win_h, rect_count, rect, rect_scores);
end
function rect_scores = vp_nonmax_custom(win_w, win_h, rect_count, rect, rect_scores)
    window_area = win_w * win_h;
    if(rect_count > 0)
        for i=1:rect_count-1
            for j=i:rect_count
                x_check = abs(rect(i,1) - rect(j,1));
                y_check = abs(rect(i,2) - rect(j,2));
                if x_check < win_w && y_check < win_h
                    crossover_area = (win_w - x_check) * (win_h - y_check);
                    persentage_crossover = crossover_area / window_area;
                    if persentage_crossover > 0.45
                        if rect_scores(i) > rect_scores(j) && rect_scores(i) ~= 0 && rect_scores(j) ~= 0
                            rect_scores(j) = 0;
                        elseif rect_scores(i) < rect_scores(j) && rect_scores(i) ~= 0 && rect_scores(j) ~= 0
                            rect_scores(i) = 0;
                        end
                    end
                end
            end
        end
    end
end                    

