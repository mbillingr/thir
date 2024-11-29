

fib 0 = 1
  | 1 = 1
  | n = (fib n - 1) + (fib n - 2);
    
main () = {
  puts(show(fib 5));
  puts "\n"
};

